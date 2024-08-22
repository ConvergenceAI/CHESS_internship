import sqlite3
import random
from typing import Union, Any, List, Dict


def execute_query(db_path: str, sql: str, fetch: Union[str, int] = "all") -> Any:
    """
    Executes an SQL query on a database and fetches results.

    Args:
        db_path (str): The path to the database file.
        sql (str): The SQL query to execute.
        fetch (Union[str, int]): How to fetch the results. Options are "all", "one", "random", or an integer.

    Returns:
        Any: The fetched results based on the fetch argument.

    Raises:
        Exception: If an error occurs during SQL execution.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            if fetch == "all":
                return cursor.fetchall()
            elif fetch == "one":
                return cursor.fetchone()
            elif fetch == "random":
                samples = cursor.fetchall()
                return random.choice(samples) if samples else []
            elif isinstance(fetch, int):
                return cursor.fetchmany(fetch)
            else:
                raise ValueError("Invalid fetch argument. Must be 'all', 'one', 'random', or an integer.")
    except Exception as e:
        raise e


def validate_sql_query(db_path: str, sql: str, max_returned_rows: int = 30) -> Dict[str, Union[str, Any]]:
    """
    Validates an SQL query by executing it and returning the result.

    Args:
        db_path (str): The path to the database file.
        sql (str): The SQL query to validate.
        max_returned_rows (int): The maximum number of rows to return.

    Returns:
        dict: A dictionary with the SQL query, result, and status.
    """
    try:
        result = execute_query(db_path, sql, fetch=max_returned_rows)
        return {"SQL": sql, "RESULT": result, "STATUS": "OK"}
    except Exception as e:
        return {"SQL": sql, "RESULT": str(e), "STATUS": "ERROR"}


def aggregate_sqls(db_path: str, sqls: List[str]) -> str:
    """
    Aggregates multiple SQL queries by validating them and clustering based on result sets.

    Args:
        db_path (str): The path to the database file.
        sqls (List[str]): A list of SQL queries to aggregate.

    Returns:
        str: The shortest SQL query from the largest cluster of equivalent queries.
    """
    results = [validate_sql_query(db_path, sql) for sql in sqls]
    clusters = {}

    # Group queries by unique result sets
    for result in results:
        if result['STATUS'] == 'OK':
            # Using a frozenset as the key to handle unhashable types like lists
            key = frozenset(tuple(row) for row in result['RESULT'])
            if key in clusters:
                clusters[key].append(result['SQL'])
            else:
                clusters[key] = [result['SQL']]

    if clusters:
        # Find the largest cluster
        largest_cluster = max(clusters.values(), key=len, default=[])
        # Select the shortest SQL query from the largest cluster
        if largest_cluster:
            return min(largest_cluster, key=len)
    return sqls[0]
