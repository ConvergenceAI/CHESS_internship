import json
from pathlib import Path
from typing import Any, Dict
import os
from tqdm import tqdm
from utils.database_utils.schema import DatabaseSchema
from utils.database_utils.schema_generator import DatabaseSchemaGenerator
from utils.database_utils.db_info import get_db_schema
from src.llm.llm_interface import UnifiedLLMInterface
from dotenv import load_dotenv
from utils.prompt import load_prompt
from utils.response_parsers import json_parser

load_dotenv()
PROMPT_PATH = os.getenv("PROMPT_ROOT_PATH") + "\column_filtering.txt"


def column_filtering(task: Any, retrieved_entities: Dict[str, Any], retrieved_context: Dict[str, Any], model: str) -> \
        Dict[str, Any]:
    """
    Filters columns based on profiles

    Args:
        task (Any): The task object containing the question and evidence.
        retrieved_entities (Dict[str, Any]): The result of the entity retrieval process
        retrieved_context (Dict[str, Any]): The result of the context retrieval process
        model (str): The LLM model used to filter columns

    Returns:
        Dict[str, Any]: A dictionary containing the updated database schema.
    """
    db_directory_path = os.getenv("DB_ROOT_PATH") + f"/{task.db_id}"
    db_path = db_directory_path + f"/{task.db_id}.sqlite"
    schema_with_descriptions = retrieved_context["schema_with_descriptions"]
    schema_with_examples = retrieved_entities["similar_values"]

    database_schema_generator = DatabaseSchemaGenerator(
        tentative_schema=DatabaseSchema.from_schema_dict(get_db_schema(db_path)),
        schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(schema_with_examples),
        schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(schema_with_descriptions),
        db_id=task.db_id,
        db_path=db_path,
        add_examples=True,
    )

    column_profiles = database_schema_generator.get_column_profiles(with_keys=True, with_references=True)

    llm = UnifiedLLMInterface()
    prompt_template = load_prompt(PROMPT_PATH)
    responses = []
    for table_name, columns in tqdm(column_profiles.items()):
        for column_name, column_profile in tqdm(columns.items()):
            prompt = prompt_template.format(COLUMN_PROFILE=column_profile, QUESTION=task.question, HINT=task.evidence)
            response = llm.generate(model, prompt)
            response = json_parser(response)
            responses.append(response)

    index = 0
    schema = {}
    for table_name, columns in column_profiles.items():
        schema[table_name] = []
        for column_name, column_profile in columns.items():
            try:
                chosen = (responses[index]["is_column_information_relevant"].lower() == "yes")
                if chosen:
                    schema[table_name].append(column_name)
            except Exception as e:
                raise e
            index += 1

    similar_columns = retrieved_entities["similar_columns"]

    # add the retrieved similar columns from the entity retrieval module to the schema
    for table_name, columns in similar_columns.items():
        target_table_name = next((t for t in schema.keys() if t.lower() == table_name.lower()), None)
        if target_table_name:
            for column in columns:
                if column.lower() not in [c.lower() for c in schema[target_table_name]]:
                    schema[target_table_name].append(column)
        else:
            schema[table_name] = columns

    # add connections to the schema
    schema_generator = DatabaseSchemaGenerator(
        tentative_schema=DatabaseSchema.from_schema_dict(schema),
        db_id=task.db_id,
        db_path=db_path,
    )
    filtered_schema = schema_generator.get_schema_with_connections()

    result = {"filtered_schema": filtered_schema}
    return result
