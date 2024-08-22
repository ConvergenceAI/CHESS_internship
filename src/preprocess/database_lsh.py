import pickle
from datasketch import MinHash, MinHashLSH
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

from utils.database_utils.execute import execute_query

TABLES_QUERY = "SELECT name FROM sqlite_master WHERE type='table';"


def _extract_unique_values(db_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Extracts unique text values from the database excluding primary keys.

    Args:
        db_path (str): The path to the SQLite database file.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary containing unique values for each table and column.
    """
    # the SQL query to extract table names from a database

    table_names = [res[0] for res in execute_query(db_path, TABLES_QUERY)]
    unique_values: Dict[str, Dict[str, List[str]]] = {}
    for table_name in tqdm(table_names):
        # must be skipped because it is not relevant (used to store information about autoincrement columns)
        if table_name == "sqlite_sequence":
            continue
        # extract column names for text and not primary ones
        columns = [col[1] for col in execute_query(db_path, f"PRAGMA table_info('{table_name}')", fetch="all") if
                   ("TEXT" in col[2] and col[5] == 0)]

        table_values: Dict[str, List[str]] = {}
        for column in columns:
            # first we must skipp some columns that are of type TEXT but in the INTEGER format (time,phone,date) and other columns irrelevant and long to be hashed (url,web,email)
            if any(keyword in column.lower() for keyword in ["url", "email", "web", "time", "phone", "date"]):
                continue

            # we must skip also the columns with long text and with no values

            sum_of_lengths, count_distinct = execute_query(db_path,
                                                           f"""
                SELECT SUM(LENGTH(unique_values)), COUNT(unique_values)
                FROM (
                    SELECT DISTINCT `{column}` AS unique_values
                    FROM `{table_name}`
                    WHERE `{column}` IS NOT NULL
                ) AS subquery
            """, fetch="one")

            if sum_of_lengths is None or count_distinct == 0:
                continue
            average_length = sum_of_lengths / count_distinct
            if average_length > 30:
                continue
            values = [str(value[0]) for value in execute_query(db_path,
                                                               f"SELECT DISTINCT `{column}` FROM `{table_name}` WHERE `{column}` IS NOT NULL")]
            table_values[column] = values
        unique_values[table_name] = table_values
    return unique_values


def _create_minhash(signature_size: int, string: str, n_gram: int) -> MinHash:
    """
    Creates a MinHash object for a given string.

    Args:
        signature_size (int): The size of the MinHash signature.
        string (str): The input string to create the MinHash for.
        n_gram (int): The n-gram size for the MinHash.

    Returns:
        MinHash: The MinHash object for the input string.
    """
    m = MinHash(num_perm=signature_size)
    # shingling
    n_grams = set(string[i:i + n_gram] for i in range(len(string) - n_gram + 1))
    # encoding
    for n_gram in n_grams:
        m.update(n_gram.encode('utf8'))
    return m


def make_lsh(unique_values: Dict[str, Dict[str, List[str]]], signature_size: int, n_gram: int, threshold: float,
             ) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
    """
    Creates a MinHash LSH from unique values.

    Args:
        unique_values (Dict[str, Dict[str, List[str]]]): The dictionary of unique values.
        signature_size (int): The size of the MinHash signature.
        n_gram (int): The n-gram size for the MinHash.
        threshold (float): The threshold for the MinHash LSH.

    Returns:
        Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The MinHash LSH object and the dictionary of MinHashes.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=signature_size)
    minhashes: Dict[str, Tuple[MinHash, str, str, str]] = {}
    try:
        unique_values_count = sum(
            len(column_values) for table_values in unique_values.values() for column_values in table_values.values())

        progress_bar = tqdm(total=unique_values_count, desc="Creating LSH")

        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():

                for i, value in enumerate(column_values):
                    minhash = _create_minhash(signature_size, value, n_gram)
                    minhash_key = f"{table_name}_{column_name}_{i}"
                    minhashes[minhash_key] = (minhash, table_name, column_name, value)
                    lsh.insert(minhash_key, minhash)

                    progress_bar.update(1)

        progress_bar.close()
    except Exception as e:
        raise e

    return lsh, minhashes


def make_db_lsh(db_directory_path: str, **kwargs: Any) -> None:
    """
    Creates a MinHash LSH for the database and saves the results.

    Args:
        db_directory_path (str): The path to the database directory.
        **kwargs (Any): Additional arguments for the LSH creation.
    """
    db_id = Path(db_directory_path).name
    preprocessed_path = Path(db_directory_path) / "preprocessed"
    preprocessed_path.mkdir(exist_ok=True)

    unique_values = _extract_unique_values(str(Path(db_directory_path) / f"{db_id}.sqlite"))

    with open(preprocessed_path / f"{db_id}_unique_values.pkl", "wb") as file:
        pickle.dump(unique_values, file)

    lsh, minhashes = make_lsh(unique_values, **kwargs)

    with open(preprocessed_path / f"{db_id}_lsh.pkl", "wb") as file:
        pickle.dump(lsh, file)
    with open(preprocessed_path / f"{db_id}_minhashes.pkl", "wb") as file:
        pickle.dump(minhashes, file)
