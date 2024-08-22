import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

load_dotenv()
import pandas as pd
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings

EMBEDDING_FUNCTION = OpenAIEmbeddings(model="text-embedding-3-large")


def make_context_vec_db(db_directory_path: str) -> None:
    """
    Creates a context vector database for the specified database directory.

    Args:
        db_directory_path (str): The path to the database directory.

    """
    table_description = load_tables_description(db_directory_path)
    docs = []

    for table_name, columns in table_description.items():
        for column_name, column_info in columns.items():
            metadata = {
                "table_name": table_name,
                "original_column_name": column_name,
                "column_name": column_info.get('column_name', ''),
                "column_description": column_info.get('column_description', ''),
                "value_description": column_info.get('value_description', '')
            }
            for key in ['column_name', 'column_description', 'value_description']:
                if column_info.get(key, '').strip():
                    docs.append(Document(page_content=column_info[key], metadata=metadata))

    vector_db_path = Path(db_directory_path) / "context_vector_db"

    if vector_db_path.exists():
        os.system(f"rm -r {vector_db_path}")

    vector_db_path.mkdir(exist_ok=True)

    Chroma.from_documents(docs, EMBEDDING_FUNCTION, persist_directory=str(vector_db_path))


def load_tables_description(db_directory_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Loads table descriptions from CSV files in the database directory.

    Args:
        db_directory_path (str): The path to the database directory.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: A dictionary containing table descriptions.
    """
    encoding_types = ['utf-8-sig', 'cp1252']
    description_path = Path(db_directory_path) / "database_description"

    if not description_path.exists():
        return {}

    table_description = {}
    for csv_file in description_path.glob("*.csv"):
        table_name = csv_file.stem.lower().strip()
        table_description[table_name] = {}

        for encoding_type in encoding_types:
            try:
                table_description_df = pd.read_csv(csv_file, index_col=False, encoding=encoding_type)
                for _, row in table_description_df.iterrows():
                    column_name = row['original_column_name']
                    expanded_column_name = row.get('column_name', '').strip() if pd.notna(
                        row.get('column_name', '')) else ""
                    column_description = row.get('column_description', '').replace('\n', ' ').replace(
                        "commonsense evidence:", "").strip() if pd.notna(row.get('column_description', '')) else ""
                    data_format = row.get('data_format', '').strip() if pd.notna(row.get('data_format', '')) else ""

                    value_description = row['value_description'].replace('\n', ' ').replace("commonsense evidence:",
                                                                                            "").strip() if pd.notna(
                        row.get('value_description', '')) else ""

                    if value_description.lower().startswith("not useful"):
                        value_description = value_description[10:].strip()

                    table_description[table_name][column_name.lower().strip()] = {
                        "original_column_name": column_name,
                        "column_name": expanded_column_name,
                        "column_description": column_description,
                        "data_format": data_format,
                        "value_description": value_description
                    }
                break
            except Exception as e:
                continue

    return table_description
