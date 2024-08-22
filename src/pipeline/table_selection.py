import os
from typing import Any, Dict, List
from src.llm.llm_interface import UnifiedLLMInterface
from utils.prompt import load_prompt
from utils.database_utils.schema import DatabaseSchema
from utils.database_utils.schema_generator import DatabaseSchemaGenerator
from dotenv import load_dotenv

from utils.response_parsers import json_parser

load_dotenv()
PROMPT_PATH = os.getenv("PROMPT_ROOT_PATH") + "\\table_selection.txt"


def table_selection(task: Any, retrieved_entities: Dict[str, Any], retrieved_context: Dict[str, Any],
                    tentative_schema: Dict[str, List[str]], model: str, num_samples=1) -> Dict[str, Any]:
    """

    Selects tables based on the task question and hint.

    Args:
        task (Any): The task object containing the question and evidence.
        retrieved_entities (Dict[str, Any]): The result of the entity retrieval process
        retrieved_context (Dict[str, Any]): The result of the context retrieval process
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        model (str): The LLM model used to select tables
        num_samples(int): The number of samples to be taken(number of repetition of the process) Default = 1

    Returns:
        Dict[str, Any]: A dictionary containing the updated tentative schema and selected tables.
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
    print(prompt)
    responses = []
    for _ in range(num_samples):
        response = llm.generate(model, prompt)
        response = json_parser(response)
        responses.append(response)

    aggregated_result = aggregate_tables(responses)

    table_names = aggregated_result["table_names"]
    result = {
        "chain_of_thought_reasoning": aggregated_result["chain_of_thought_reasoning"],
        "selected_tables": table_names,
    }

    tentative_schema = {
        table_name: tentative_schema.get(table_name, [])
        for table_name in table_names
    }

    similar_columns = retrieved_entities["similar_columns"]
    # add the retrieved similar columns from the entity retrieval module to the schema
    for table_name, columns in similar_columns.items():
        target_table_name = next((t for t in tentative_schema.keys() if t.lower() == table_name.lower()), None)
        if target_table_name:
            for column in columns:
                if column.lower() not in [c.lower() for c in tentative_schema[target_table_name]]:
                    tentative_schema[target_table_name].append(column)
        else:
            tentative_schema[table_name] = columns
    # add connections to the schema
    schema_generator = DatabaseSchemaGenerator(
        tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
        db_id=task.db_id,
        db_path=db_path,
    )
    tentative_schema = schema_generator.get_schema_with_connections()

    result = {"tentative_schema": tentative_schema, **result}
    return result


def aggregate_tables(tables_dicts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Aggregates tables from multiple responses and consolidates reasoning.

    Args:
        tables_dicts (List[Dict[str, Any]]): List of dictionaries containing table names and reasoning.

    Returns:
        Dict[str, List[str]]: Aggregated result with unique table names and consolidated reasoning.
    """
    tables = []
    chain_of_thoughts = []
    for table_dict in tables_dicts:
        chain_of_thoughts.append(table_dict.get("chain_of_thought_reasoning", ""))
        response_tables = table_dict.get("table_names", [])
        for table in response_tables:
            if table.lower() not in [t.lower() for t in tables]:
                tables.append(table)

    aggregated_chain_of_thoughts = "\n----\n".join(chain_of_thoughts)
    aggregation_result = {
        "table_names": tables,
        "chain_of_thought_reasoning": aggregated_chain_of_thoughts,
    }
    return aggregation_result
