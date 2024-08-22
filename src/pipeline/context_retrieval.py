import os
from typing import Dict, Any, List
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
from src.preprocess.vector_database import EMBEDDING_FUNCTION


def context_retrieval(task: Any, keywords: List[str], top_k: int) -> Dict[str, Any]:
    """
    Retrieves context information based on the task's question and evidence.

    Args:
        task (Any): The task object containing the question and evidence.
        keywords (List[str]): The keywords extracted from the evidence and question
        top_k (int): The number of top similar documents to retrieve
    Returns:
        Dict[str, Any]: A dictionary containing the schema with descriptions.

    """

    retrieved_columns = _find_most_similar_columns(
        question=task.question,
        evidence=task.evidence,
        db_id=task.db_id,
        keywords=keywords,
        top_k=top_k
    )
    # Formats retrieved descriptions by removing the score key.
    for column_descriptions in retrieved_columns.values():
        for column_info in column_descriptions.values():
            column_info.pop("score", None)
    result = {"schema_with_descriptions": retrieved_columns}

    return result


def _find_most_similar_columns(question: str, evidence: str, db_id: str, keywords: List[str], top_k: int) -> Dict[
    str, Dict[str, Dict[str, str]]]:
    """
    Finds the most similar columns based on the question and evidence.

    Args:
        question (str): The question string.
        evidence (str): The evidence string.
        db_id (str): The database id
        keywords (List[str]): The list of keywords.
        top_k (int): The number of top similar columns to retrieve.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: A dictionary containing the most similar columns with descriptions.
    """
    tables_with_descriptions = {}

    for keyword in keywords:
        question_based_query = f"{question} {keyword}"
        evidence_based_query = f"{evidence} {keyword}"

        retrieved_question_based_query = query_vector_db(db_id, question_based_query, top_k=top_k)
        retrieved_evidence_based_query = query_vector_db(db_id, evidence_based_query, top_k=top_k)

        tables_with_descriptions = _add_description(tables_with_descriptions, retrieved_question_based_query)
        tables_with_descriptions = _add_description(tables_with_descriptions, retrieved_evidence_based_query)

    return tables_with_descriptions


def _add_description(tables_with_descriptions: Dict[str, Dict[str, Dict[str, str]]],
                     retrieved_descriptions: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[
    str, Dict[str, Dict[str, str]]]:
    """
    Adds descriptions to tables from retrieved descriptions.

    Args:
        tables_with_descriptions (Dict[str, Dict[str, Dict[str, str]]]): The current tables with descriptions.
        retrieved_descriptions (Dict[str, Dict[str, Dict[str, str]]]): The retrieved descriptions.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: The updated tables with descriptions.
    """
    for table_name, column_descriptions in retrieved_descriptions.items():
        if table_name not in tables_with_descriptions:
            tables_with_descriptions[table_name] = {}
        for column_name, description in column_descriptions.items():
            if (column_name not in tables_with_descriptions[table_name] or
                    description["score"] > tables_with_descriptions[table_name][column_name]["score"]):
                tables_with_descriptions[table_name][column_name] = description
    return tables_with_descriptions


def query_vector_db(db_id: str, query: str, top_k: int) -> Dict[str, Dict[str, dict]]:
    """
    Queries the vector database for the most relevant documents based on the query.

    Args:
        db_id (str): The database name to query.
        query (str): The query string to search for.
        top_k (int): The number of top results to return.

    Returns:
        Dict[str, Dict[str, dict]]: A dictionary containing table descriptions with their column details and scores.
    """
    vector_db_path = os.getenv("DB_ROOT_PATH") + f"/{db_id}/context_vector_db"
    vector_db = Chroma(persist_directory=str(vector_db_path), embedding_function=EMBEDDING_FUNCTION)
    table_description = {}

    try:
        relevant_docs_score = vector_db.similarity_search_with_score(query, k=top_k)
    except Exception as e:
        raise e

    for doc, score in relevant_docs_score:
        metadata = doc.metadata
        table_name = metadata["table_name"]
        original_column_name = metadata["original_column_name"].strip()
        column_name = metadata["column_name"].strip()
        column_description = metadata["column_description"].strip()
        value_description = metadata["value_description"].strip()

        if table_name not in table_description:
            table_description[table_name] = {}

        if original_column_name not in table_description[table_name]:
            table_description[table_name][original_column_name] = {
                "column_name": column_name,
                "column_description": column_description,
                "value_description": value_description,
                "score": score
            }

    return table_description
