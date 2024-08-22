import difflib
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv

load_dotenv()
from datasketch import MinHash, MinHashLSH

from src.preprocess.database_lsh import _create_minhash
from src.embedding.embedding_interface import UnifiedEmbeddingInterface
from utils.database_utils.db_info import get_db_schema

EMBEDDING = UnifiedEmbeddingInterface()


def entity_retrieval(task: Any, embedding_model_name: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Retrieves entities and columns similar to given keywords from the task.

    Args:
        task (Any): The task object containing the evidence and question.
        embedding_model_name (str): The name of the embedding model used in the entity retrieval process
        keywords (List[str]): The keywords extracted from the evidence and question

    Returns:
        Dict[str, Any]: A dictionary containing similar columns and values.
    """

    db_path = os.getenv("DB_ROOT_PATH") + "/" + task.db_id
    similar_columns = get_similar_columns(db_path=db_path, keywords=keywords, question=task.question,
                                          hint=task.evidence, embedding_model_name=embedding_model_name)
    result = {"similar_columns": similar_columns}

    similar_values = get_similar_entities(embedding_model_name=embedding_model_name, db_path=db_path,
                                          keywords=keywords)
    result["similar_values"] = similar_values

    return result


### Column similarity ###
def get_similar_columns(db_path: str, keywords: List[str], question: str, hint: str, embedding_model_name: str) -> Dict[
    str, List[str]]:
    """
    Finds columns similar to given keywords based on question and hint.

    Args:
        keywords (List[str]): The list of keywords.
        question (str): The question string.
        hint (str): The hint string.

    Returns:
        Dict[str, List[str]]: A dictionary mapping table names to lists of similar column names.
    """
    selected_columns = {}
    for keyword in keywords:
        similar_columns = _get_similar_column_names(db_path=db_path, keyword=keyword, question=question, hint=hint,
                                                    embedding_model_name=embedding_model_name)
        for table_name, column_name in similar_columns:
            selected_columns.setdefault(table_name, []).append(column_name)
    return selected_columns


def _get_similar_column_names(db_path: str, keyword: str, question: str, hint: str, embedding_model_name: str) -> List[
    Tuple[str, str]]:
    """
    Finds column names similar to a keyword.

    Args:
        keyword (str): The keyword to find similar columns for.
        question (str): The question string.
        hint (str): The hint string.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing table and column names.
    """
    keyword = keyword.strip()
    potential_column_names = [keyword]

    column, value = _column_value(keyword)
    if column:
        potential_column_names.append(column)

    potential_column_names.extend(_extract_paranthesis(keyword))

    if " " in keyword:
        potential_column_names.extend(part.strip() for part in keyword.split())
    db_name = db_path.split("/")[-1]
    db_path = db_path + "/" + db_name + ".sqlite"
    schema = get_db_schema(db_path)

    similar_column_names = []
    for table, columns in schema.items():
        for column in columns:
            for potential_column_name in potential_column_names:
                if _does_keyword_match_column(potential_column_name, column):
                    similarity_score = EMBEDDING.compute_similarities(embedding_model_name, f"`{table}`.`{column}`",
                                                                      [f"{question} {hint}"])
                    similar_column_names.append((table, column, similarity_score))

    similar_column_names.sort(key=lambda x: x[2], reverse=True)
    return [(table, column) for table, column, _ in similar_column_names[:1]]


def _extract_paranthesis(string: str) -> List[str]:
    """
    Extracts strings within parentheses from a given string.

    Args:
        string (str): The string to extract from.

    Returns:
        List[str]: A list of strings within parentheses.
    """
    paranthesis_matches = []
    open_paranthesis = []
    for i, char in enumerate(string):
        if char == "(":
            open_paranthesis.append(i)
        elif char == ")" and open_paranthesis:
            start = open_paranthesis.pop()
            found_string = string[start:i + 1]
            if found_string:
                paranthesis_matches.append(found_string)
    return paranthesis_matches


def _does_keyword_match_column(keyword: str, column_name: str, threshold: float = 0.9) -> bool:
    """
    Checks if a keyword matches a column name based on similarity.

    Args:
        keyword (str): The keyword to match.
        column_name (str): The column name to match against.
        threshold (float, optional): The similarity threshold. Defaults to 0.9.

    Returns:
        bool: True if the keyword matches the column name, False otherwise.
    """
    keyword = keyword.lower().replace(" ", "").replace("_", "").rstrip("s")
    column_name = column_name.lower().replace(" ", "").replace("_", "").rstrip("s")

    similarity = difflib.SequenceMatcher(None, column_name, keyword).ratio()
    return similarity >= threshold


### Entity similarity ###

def get_similar_entities(embedding_model_name: str, db_path: str, keywords: List[str]) -> Dict[
    str, Dict[str, List[str]]]:
    """
    Retrieves similar entities from the database based on keywords.

    Args:
        keywords (List[str]): The list of keywords.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary mapping table and column names to similar entities.
    """

    lsh, minhashes = _load_db_lsh(db_path)
    selected_values = {}

    for keyword in keywords:
        keyword = keyword.strip()
        to_search_values = [keyword]
        if (" " in keyword) and ("=" not in keyword):
            for i in range(len(keyword)):
                if keyword[i] == " ":
                    first_part = keyword[:i]
                    second_part = keyword[i + 1:]
                    if first_part not in to_search_values:
                        to_search_values.append(first_part)
                    if second_part not in to_search_values:
                        to_search_values.append(second_part)

        to_search_values.sort(key=len, reverse=True)
        hint_column, hint_value = _column_value(keyword)
        if hint_value:
            to_search_values = [hint_value, *to_search_values]

        for ts in to_search_values:
            unique_similar_values = _query_lsh(lsh, minhashes, keyword=ts, signature_size=20, top_n=10)
            similar_values = _get_similar_entities_to_keyword(embedding_model_name, ts, unique_similar_values)

            for table_name, column_values in similar_values.items():
                for column_name, entities in column_values.items():
                    if entities:
                        selected_values.setdefault(table_name, {}).setdefault(column_name, []).extend(
                            [(ts, value, edit_distance, embedding) for ts, value, edit_distance, embedding in entities])

    for table_name, column_values in selected_values.items():
        for column_name, values in column_values.items():
            max_edit_distance = max(values, key=lambda x: x[2])[2]
            selected_values[table_name][column_name] = list(set(
                value for _, value, edit_distance, _ in values if edit_distance == max_edit_distance
            ))
    return selected_values


def _get_similar_entities_to_keyword(embedding_model_name: str, keyword: str,
                                     unique_values: Dict[str, Dict[str, List[str]]]) -> Dict[
    str, Dict[str, List[Tuple[str, str, float, float]]]]:
    """
    Finds entities similar to a keyword in the database.

    Args:
        keyword (str): The keyword to find similar entities for.
        unique_values (Dict[str, Dict[str, List[str]]]): The dictionary of unique values from the database.

    Returns:
        Dict[str, Dict[str, List[Tuple[str, str, float, float]]]]: A dictionary mapping table and column names to similar entities.
    """
    return {
        table_name: {
            column_name: _get_similar_values(embedding_model_name, keyword, values)
            for column_name, values in column_values.items()
        }
        for table_name, column_values in unique_values.items()
    }


def _get_similar_values(embedding_model_name: str, target_string: str, values: List[str]) -> List[
    Tuple[str, str, float, float]]:
    """
    Finds values similar to the target string based on edit distance and embedding similarity.

    Args:
        target_string (str): The target string to compare against.
        values (List[str]): The list of values to compare.

    Returns:
        List[Tuple[str, str, float, float]]: A list of tuples containing the target string, value, edit distance, and embedding similarity.
    """

    # hyperparameters
    edit_distance_threshold = 0.3
    top_k_edit_distance = 5
    embedding_similarity_threshold = 0.6
    top_k_embedding = 1

    # extract the values with edit distance above the threshold
    edit_distance_similar_values = [
        (value, difflib.SequenceMatcher(None, value.lower(), target_string.lower()).ratio())
        for value in values
        if difflib.SequenceMatcher(None, value.lower(), target_string.lower()).ratio() >= edit_distance_threshold
    ]

    # sort them in decreasing order and extract the top_k_edit distance
    edit_distance_similar_values.sort(key=lambda x: x[1], reverse=True)
    edit_distance_similar_values = edit_distance_similar_values[:top_k_edit_distance]

    # compute the embedding similarities of remaining values
    similarities = EMBEDDING.compute_similarities(embedding_model_name, target_string,
                                                  [value for value, _ in edit_distance_similar_values])
    # extract the values with embedding similarities above the threshold
    embedding_similar_values = [
        (target_string, edit_distance_similar_values[i][0], edit_distance_similar_values[i][1], similarities[i])
        for i in range(len(edit_distance_similar_values))
        if similarities[i] >= embedding_similarity_threshold
    ]

    embedding_similar_values.sort(key=lambda x: x[2], reverse=True)
    return embedding_similar_values[:top_k_embedding]


def _column_value(string: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Splits a string into column and value parts if it contains '='.

    Args:
        string (str): The string to split.

    Returns:
        Tuple[Optional[str], Optional[str]]: The column and value parts.
    """
    if "=" in string:
        left_equal = string.find("=")
        first_part = string[:left_equal].strip()
        second_part = string[left_equal + 1:].strip() if len(string) > left_equal + 1 else None
        return first_part, second_part
    return None, None


### Database value similarity retrieval
def _jaccard_similarity(m1: MinHash, m2: MinHash) -> float:
    """
    Computes the Jaccard similarity between two MinHash objects.

    Args:
        m1 (MinHash): The first MinHash object.
        m2 (MinHash): The second MinHash object.

    Returns:
        float: The Jaccard similarity between the two MinHash objects.
    """
    return m1.jaccard(m2)


def _load_db_lsh(db_directory_path: str) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
    """
    Loads the LSH and MinHashes from the preprocessed files for the specified database.

    Args:
        db_directory_path (str): The path to the database directory.

    Returns:
        Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The LSH object and the dictionary of MinHashes.

    Raises:
        Exception: If there is an error loading the LSH or MinHashes.
    """
    db_id = Path(db_directory_path).name
    try:
        with open(Path(db_directory_path) / "preprocessed" / f"{db_id}_lsh.pkl", "rb") as file:
            lsh = pickle.load(file)
        with open(Path(db_directory_path) / "preprocessed" / f"{db_id}_minhashes.pkl", "rb") as file:
            minhashes = pickle.load(file)
        return lsh, minhashes
    except Exception as e:
        raise e


def _query_lsh(lsh: MinHashLSH, minhashes: Dict[str, Tuple[MinHash, str, str, str]], keyword: str,
               signature_size: int = 20, n_gram: int = 3, top_n: int = 10) -> Dict[str, Dict[str, List[str]]]:
    """
    Queries the LSH for similar values to the given keyword and returns the top_n results.

    Args:
        lsh (MinHashLSH): The LSH object.
        minhashes (Dict[str, Tuple[MinHash, str, str, str]]): The dictionary of MinHashes.
        keyword (str): The keyword to search for.
        signature_size (int, optional): The size of the MinHash signature.
        n_gram (int, optional): The n-gram size for the MinHash.
        top_n (int, optional): The number of top results to return.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary containing the top similar values.
    """
    # create the minhash of the searched keyword
    query_minhash = _create_minhash(signature_size, keyword, n_gram)
    # retrieve the keys of similar values of lsh as the query_minhash
    results = lsh.query(query_minhash)  # returns a list of keys
    # compute the jaccard similarities for the retrieved minhashes with the query_minhash
    similarities = [(result, _jaccard_similarity(query_minhash, minhashes[result][0])) for result in results]

    # sort in descending order based on the jaccard similarity and get the top_n similar values
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    similar_values_trimmed: Dict[str, Dict[str, List[str]]] = {}
    for hash_key, similarity in similarities:
        table_name, column_name, value = minhashes[hash_key][1:]
        if table_name not in similar_values_trimmed:
            similar_values_trimmed[table_name] = {}
        if column_name not in similar_values_trimmed[table_name]:
            similar_values_trimmed[table_name][column_name] = []
        similar_values_trimmed[table_name][column_name].append(value)

    return similar_values_trimmed
