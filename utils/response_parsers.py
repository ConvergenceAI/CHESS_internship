import ast
import json
import re


def list_parser(response: str):
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0]
    response = re.sub(r"^\s+", "", response)
    response = ast.literal_eval(response)
    return response


def json_parser(response: str):
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    response = re.sub(r"^\s+", "", response)
    response = response.replace("\n", " ").replace("\t", " ")
    response = json.loads(response)
    return response


def candidate_generation_parser(response: str):
    if "```sql" in response:
        response = response.split("```sql")[1].split("```")[0]
        output = re.sub(r"^\s+", "", response)
        return {"SQL": output}

    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
        response = re.sub(r"^\s+", "", response)
        response = response.replace("\n", " ").replace("\t", " ")
        response = json.loads(response)
        return response
    return json.loads(response)
