import os
from typing import Any, Dict, List
from src.llm.llm_interface import UnifiedLLMInterface
from utils.prompt import load_prompt
from utils.database_utils.schema import DatabaseSchema
from utils.database_utils.schema_generator import DatabaseSchemaGenerator
from dotenv import load_dotenv

from utils.response_parsers import json_parser

load_dotenv()
PROMPT_PATH = os.getenv("PROMPT_ROOT_PATH") + "\\column_selection.txt"


def column_selection(task: Any, retrieved_entities: Dict[str, Any], retrieved_context: Dict[str, Any],
                     tentative_schema: Dict[str, List[str]], model: str, num_samples=1) -> Dict[str, Any]:
    """
    Selects columns based on the task question and hint

    Args:
        task (Any): The task object containing the question and evidence.
        retrieved_entities (Dict[str, Any]): The result of the entity retrieval process
        retrieved_context (Dict[str, Any]): The result of the context retrieval process
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        model (str): The LLM model used to select columns
        num_samples(int): The number of samples to be taken(number of repetition of the process) Default = 1

    Returns:
        Dict[str, Any]: A dictionary containing the updated tentative schema and selected columns.
    """
    db_directory_path = os.getenv("DB_ROOT_PATH") + f"/{task.db_id}"
    db_path = db_directory_path + f"/{task.db_id}.sqlite"
    schema_with_examples = retrieved_entities["similar_values"]
    schema_with_descriptions = retrieved_context["schema_with_descriptions"]

    schema_generator = DatabaseSchemaGenerator(
        tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
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
    responses = []
    for _ in range(num_samples):
        response = llm.generate(model, prompt)
        response = json_parser(response)
        responses.append(response)

    aggregated_result = aggregate_columns(responses, list(tentative_schema.keys()))
    column_names = aggregated_result
    chain_of_thought_reasoning = aggregated_result.pop("chain_of_thought_reasoning")
    result = {
        "selected_schema": column_names,
        "chain_of_thought_reasoning": chain_of_thought_reasoning
    }
    return result


def aggregate_columns(columns_dicts: List[Dict[str, Any]], selected_tables: List[str]) -> Dict[str, List[str]]:
    """
    Aggregates columns from multiple responses and consolidates reasoning.

    Args:
        columns_dicts (List[Dict[str, Any]]): List of dictionaries containing column names and reasoning.
        selected_tables (List[str]): List of selected tables.

    Returns:
        Dict[str, List[str]]: Aggregated result with unique column names and consolidated reasoning.
    """
    dict_cot = ""
    columns = {}
    chain_of_thoughts = []
    for column_dict in columns_dicts:
        valid_column_dict = False
        for key, value in column_dict.items():
            if key == "chain_of_thought_reasoning":
                dict_cot = value
            else:  # key is table name
                table_name = key
                if table_name.startswith("`"):
                    table_name = table_name[1:-1]
                column_names = value
                if table_name.lower() in [t.lower() for t in selected_tables]:
                    for column_name in column_names:
                        if column_name.startswith("`"):
                            column_name = column_name[1:-1]
                        if table_name not in columns:
                            columns[table_name] = []
                        if column_name.lower() not in [col.lower() for col in columns[table_name]]:
                            columns[table_name].append(column_name)
                        valid_column_dict = True
        if valid_column_dict:
            chain_of_thoughts.append(dict_cot)

    aggregated_chain_of_thoughts = "\n----\n".join(chain_of_thoughts)
    aggregation_result = columns
    aggregation_result["chain_of_thought_reasoning"] = aggregated_chain_of_thoughts
    return aggregation_result
