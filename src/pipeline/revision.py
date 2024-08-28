import difflib
import os
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

from src.llm.llm_interface import UnifiedLLMInterface
from utils.database_utils.db_info import get_db_schema
from utils.database_utils.execute import validate_sql_query, aggregate_sqls
from utils.database_utils.schema import DatabaseSchema
from utils.database_utils.schema_generator import DatabaseSchemaGenerator
from utils.prompt import load_prompt
from utils.response_parsers import json_parser
from utils.sql_query_parsers import get_sql_condition_literals

load_dotenv()

PROMPT_PATH = os.getenv("PROMPT_ROOT_PATH") + "\\revision.txt"


def revision(task: Any, retrieved_entities: Dict[str, Any], retrieved_context: Dict[str, Any],
             generated_candidate: Dict[str, Any], model: str, num_samples=1) -> Dict[str, Any]:
    """
    Revises the predicted SQL query based on task evidence and schema information.

    Args:
        task (Any): The task object containing the question and evidence.
        retrieved_entities (Dict[str, Any]): The result of the entity retrieval process
        retrieved_context (Dict[str, Any]): The result of the context retrieval process
        generated_candidate(Dict[str,Any]): The result of the candidate generation process
        model (str): The LLM model used to revise the predicted sql query
        num_samples(int): The number of samples to be taken(number of repetition of the process) Default = 1

    Returns:
        Dict[str, Any]: A dictionary containing the revised SQL query and reasoning.
    """
    schema_with_examples = retrieved_entities["similar_values"]
    schema_with_descriptions = retrieved_context["schema_with_descriptions"]
    db_directory_path = os.getenv("DB_ROOT_PATH") + f"/{task.db_id}"
    db_path = db_directory_path + f"/{task.db_id}.sqlite"

    complete_schema = get_db_schema(db_path)
    schema_generator = DatabaseSchemaGenerator(
        tentative_schema=DatabaseSchema.from_schema_dict(complete_schema),
        schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(
            schema_with_examples) if schema_with_examples else None,
        schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(
            schema_with_descriptions) if schema_with_descriptions else None,
        db_id=task.db_id,
        db_path=db_path,
    )
    # The complete database schema
    schema_string = schema_generator.generate_schema_string(include_value_description=True)

    predicted_query = generated_candidate["SQL"]

    try:
        query_result = validate_sql_query(db_path=db_path, sql=predicted_query)['RESULT']
    except Exception as e:
        raise e

    try:
        missing_entities = find_wrong_entities(db_path, predicted_query, schema_with_examples)
    except Exception as e:
        raise e

    llm = UnifiedLLMInterface()
    prompt_template = load_prompt(PROMPT_PATH)
    prompt = prompt_template.format(DATABASE_SCHEMA=schema_string, QUESTION=task.question, HINT=task.evidence,
                                    MISSING_ENTITIES=missing_entities, SQL=predicted_query, QUERY_RESULT=query_result)
    responses = []
    for _ in range(num_samples):
        response = llm.generate(model, prompt)
        response = json_parser(response)
        responses.append(response)

    revised_sqls = [res["revised_SQL"] for res in responses]
    revised_sql = aggregate_sqls(db_path=db_path, sqls=revised_sqls)
    chosen_res = next(res for res in responses if res["revised_SQL"] == revised_sql)

    result = {
        "chain_of_thought_reasoning": chosen_res.get("chain_of_thought_reasoning", ""),
        "SQL": chosen_res["revised_SQL"],
    }
    return result


def find_wrong_entities(db_path: str, sql: str, similar_values: Dict[str, Dict[str, List[str]]],
                        similarity_threshold: float = 0.4) -> str:
    """
    Finds and returns a string listing entities in the SQL that do not match the database schema.

    Args:
        db_path (str): Path to the database used in the task
        sql (str): The SQL query to check.
        similar_values (Dict[str, Dict[str, List[str]]]): Dictionary of similar values for columns.
        similarity_threshold (float, optional): The similarity threshold for matching values. Defaults to 0.4.

    Returns:
        str: A string listing the mismatched entities and suggestions.
    """
    wrong_entities = ""
    used_entities = get_sql_condition_literals(db_path, sql)
    similar_values_database_schema = DatabaseSchema.from_schema_dict_with_examples(similar_values)

    for table_name, column_info in used_entities.items():
        for column_name, column_values in column_info.items():
            target_column_info = similar_values_database_schema.get_column_info(table_name, column_name)
            if not target_column_info:
                continue
            for value in column_values:
                column_similar_values = target_column_info.examples
                if value not in column_similar_values:
                    most_similar_entity, similarity = _find_most_syntactically_similar_value(value,
                                                                                             column_similar_values)
                    if similarity > similarity_threshold:
                        wrong_entities += f"Column {column_name} in table {table_name} does not contain the value '{value}'. The correct value is '{most_similar_entity}'.\n"

    for used_table_name, used_column_info in used_entities.items():
        for used_column_name, used_values in used_column_info.items():
            for used_value in used_values:
                for table_name, column_info in similar_values.items():
                    for column_name, column_values in column_info.items():
                        if (used_value in column_values) and (column_name.lower() != used_column_name.lower()):
                            wrong_entities += f"Value {used_value} that you used in the query appears in the column {column_name} of table {table_name}.\n"
    return wrong_entities


def _find_most_syntactically_similar_value(target_value: str, candidate_values: List[str]) -> Tuple[str, float]:
    """
    Finds the most syntactically similar value to the target value from the candidate values.

    Args:
        target_value (str): The target value to match.
        candidate_values (List[str]): The list of candidate values.

    Returns:
        Tuple[str, float]: The most similar value and the similarity score.
    """
    most_similar_entity = max(candidate_values,
                              key=lambda value: difflib.SequenceMatcher(None, value, target_value).ratio(),
                              default=None)
    max_similarity = difflib.SequenceMatcher(None, most_similar_entity,
                                             target_value).ratio() if most_similar_entity else 0
    return most_similar_entity, max_similarity
