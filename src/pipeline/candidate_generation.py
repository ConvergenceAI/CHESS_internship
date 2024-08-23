import os
from typing import Any, Dict, List
from src.llm.llm_interface import UnifiedLLMInterface
from utils.database_utils.execute import aggregate_sqls
from utils.prompt import load_prompt
from utils.database_utils.schema import DatabaseSchema
from utils.database_utils.schema_generator import DatabaseSchemaGenerator
from dotenv import load_dotenv

from utils.response_parsers import candidate_generation_parser

load_dotenv()
PROMPT_PATH = os.getenv("PROMPT_ROOT_PATH") + "\candidate_generation.txt"


def candidate_generation(task: Any, retrieved_entities: Dict[str, Any], retrieved_context: Dict[str, Any],
                         selected_schema: Dict[str, List[str]], model: str, num_samples=1) -> Dict[str, Any]:
    """
    Generates candidate SQL queries based on the task's question and evidence.

    Args:
        task (Any): The task object containing the question and evidence.
        retrieved_entities (Dict[str, Any]): The result of the entity retrieval process
        retrieved_context (Dict[str, Any]): The result of the context retrieval process
        selected_schema (Dict[str, List[str]]): The selected database schema.
        model (str): The LLM model used to generate the candidate sql query
        num_samples(int): The number of samples to be taken(number of repetition of the process) Default = 1

    Returns:
        Dict[str, Any]: A dictionary containing the best SQL query result.
    """

    schema_with_examples = retrieved_entities["similar_values"]
    schema_with_descriptions = retrieved_context["schema_with_descriptions"]
    db_directory_path = os.getenv("DB_ROOT_PATH") + f"/{task.db_id}"
    db_path = db_directory_path + f"/{task.db_id}.sqlite"

    schema_generator = DatabaseSchemaGenerator(
        tentative_schema=DatabaseSchema.from_schema_dict(selected_schema),
        schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(
            schema_with_examples) if schema_with_examples else None,
        schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(
            schema_with_descriptions) if schema_with_descriptions else None,
        db_id=task.db_id,
        db_path=db_path,
    )
    schema_string = schema_generator.generate_schema_string(include_value_description=True)

    llm = UnifiedLLMInterface()
    prompt_template = load_prompt(PROMPT_PATH)
    prompt = prompt_template.format(DATABASE_SCHEMA=schema_string, QUESTION=task.question, HINT=task.evidence)
    # print(prompt)

    responses = []
    for _ in range(num_samples):
        response = llm.generate(model, prompt)
        response = candidate_generation_parser(response)
        responses.append(response)
    sqls = [res["SQL"] for res in responses]
    sql = aggregate_sqls(db_path, sqls)
    result = next(res for res in responses if res["SQL"] == sql)
    return result
