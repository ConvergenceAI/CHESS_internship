from typing import Any, Dict

from utils.database_utils.execute import compare_sqls
import os
from dotenv import load_dotenv

load_dotenv()


def evaluation(task: Any, generated_candidate: Dict[str, Any], revised_candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the predicted SQL queries against the ground truth SQL query.

    Args:
        task (Any): The task object containing the question and evidence.
        generated_candidate(Dict[str,Any]): The result of the candidate generation process
        revised_candidate (Dict[str, Any]): The result of the revision process

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation results.
    """
    db_directory_path = os.getenv("DB_ROOT_PATH") + f"/{task.db_id}"
    db_path = db_directory_path + f"/{task.db_id}.sqlite"

    ground_truth_sql = task.SQL
    to_evaluate = {
        "candidate_generation": generated_candidate,
        "revision": revised_candidate
    }
    result = {}

    for evaluation_for, result in to_evaluate.items():
        predicted_sql = "--"
        evaluation_result = {}

        try:
            predicted_sql = result["SQL"]
            response = compare_sqls(db_path,
                                    predicted_sql=predicted_sql,
                                    ground_truth_sql=ground_truth_sql,
                                    )

            evaluation_result.update({
                "exec_res": response["exec_res"],
                "exec_err": response["exec_err"],
            })

        except Exception as e:
            evaluation_result.update({
                "exec_res": "error",
                "exec_err": str(e),
            })

        evaluation_result.update({
            "Question": task.question,
            "Evidence": task.evidence,
            "GOLD_SQL": ground_truth_sql,
            "PREDICTED_SQL": predicted_sql
        })
        result[evaluation_for] = evaluation_result

    return result
