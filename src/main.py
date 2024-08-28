import json
import os
from datetime import datetime
from typing import Any, Dict, List

from src.pipeline.candidate_generation import candidate_generation
from src.pipeline.column_filtering import column_filtering
from src.pipeline.column_selection import column_selection
from src.pipeline.context_retrieval import context_retrieval
from src.pipeline.entity_retrieval import entity_retrieval
from src.pipeline.evaluation import evaluation
from src.pipeline.keyword_extraction import keyword_extraction
from src.pipeline.revision import revision
from src.pipeline.table_selection import table_selection
from src.pipeline_config import PIPELINE_CONFIG
from utils.task import Task

data_path = "C:\\Users/yousf\Bureau\ConvergenceAI\CHESS_Impl\data\\test\subsampled_test.json"


def run_task(task: Any) -> Dict[str, Any]:
    """
     runs the CHESS pipeline on a single task

     Args:
         task (Any): the task object

     Returns:
        Dict[str, Any]: A dictionary containing the results of every module in the pipeline.
    """
    # Step 1 : Keyword Extraction
    print("Running Keyword Extraction module...")
    keyword_extraction_result = keyword_extraction(
        task=task,
        model=PIPELINE_CONFIG["keyword_extraction"]["model"]
    )

    # Step 2 : Entity Retrieval
    print("Running Entity Retrival module...")
    entity_retrieval_result = entity_retrieval(
        task=task,
        embedding_model_name=PIPELINE_CONFIG["entity_retrieval"]["embedding_model_name"],
        keywords=keyword_extraction_result
    )

    # Step 3 : Context Retrieval
    print("Running Context Retrieval module...")
    context_retrieval_result = context_retrieval(
        task=task,
        keywords=keyword_extraction_result,
        top_k=PIPELINE_CONFIG["context_retrieval"]["top_k"]
    )

    # Remove Column filtering module from pipeline because even using open source model it will not work (Too many requests)

    # # Step 4 : Column Filtering
    # print("Running Column Filtering module...")
    # column_filtering_result = column_filtering(
    #     task=task,
    #     model=PIPELINE_CONFIG["column_filtering"]["model"],
    #     retrieved_entities=entity_retrieval_result,
    #     retrieved_context=context_retrieval_result
    # )

    # Step 5 : Table Selection
    print("Running Table Selection module...")
    table_selection_result = table_selection(
        task=task,
        retrieved_entities=entity_retrieval_result,
        retrieved_context=context_retrieval_result,
        tentative_schema=None,
        model=PIPELINE_CONFIG["table_selection"]["model"],
        num_samples=PIPELINE_CONFIG["table_selection"]["num_samples"]
    )

    # Step 6 : Column Selection
    print("Running Column Selection module...")
    column_selection_result = column_selection(
        task=task,
        retrieved_entities=entity_retrieval_result,
        retrieved_context=context_retrieval_result,
        tentative_schema=table_selection_result["tentative_schema"],
        model=PIPELINE_CONFIG["column_selection"]["model"],
        num_samples=PIPELINE_CONFIG["column_selection"]["num_samples"]
    )

    # Step 7 : Candidate Generation
    print("Running Candidate Generation module...")
    candidate_generation_result = candidate_generation(
        task=task,
        retrieved_entities=entity_retrieval_result,
        retrieved_context=context_retrieval_result,
        selected_schema=column_selection_result["selected_schema"],
        model=PIPELINE_CONFIG["candidate_generation"]["model"],
        num_samples=PIPELINE_CONFIG["candidate_generation"]["num_samples"]
    )

    # Step 8 : Revision
    print("Running Revision module...")
    revision_result = revision(
        task=task,
        retrieved_entities=entity_retrieval_result,
        retrieved_context=context_retrieval_result,
        generated_candidate=candidate_generation_result,
        model=PIPELINE_CONFIG["revision"]["model"],
        num_samples=PIPELINE_CONFIG["revision"]["num_samples"]
    )

    # Step 9 : Evaluation
    print("Running Evaluation module...")
    evaluation_result = evaluation(
        task=task,
        generated_candidate=candidate_generation_result,
        revised_candidate=revision_result
    )
    return {
        "keyword_extraction": keyword_extraction_result,
        "entity_retrieval": entity_retrieval_result,
        "context_retrieval": context_retrieval_result,
        "column_filtering": "Removed",
        "table_selection": table_selection_result,
        "column_selection": column_selection_result,
        "candidate_generation": candidate_generation_result,
        "revision": revision_result,
        "evaluation": evaluation_result
    }


def load_tasks(path: str) -> List[Dict[str, Any]]:
    """
    Load the tasks from the given JSON data file.

    Args:
        path (str): Path to the JSON file containing the tasks.

    Returns:
        List[Dict[str, Any]]: List of tasks.
    """
    with open(path, "r") as file:
        tasks = json.load(file)
    return tasks


def store_results(result: Dict[str, Any], output_dir: str, task_id: int):
    """
    Store the result of a task in the specified output directory.

    Args:
        result (Dict[str, Any]): The result to store.
        output_dir (str): The directory where results will be stored.
        task_id (str): Unique identifier for the task.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_path = os.path.join(output_dir, f"{task_id}_result.json")
    with open(result_path, "w") as file:
        json.dump(result, file, indent=4)
    print(f"Results for task {task_id} stored in {result_path}")


def calculate_accuracy(results: List[Dict[str, Any]]) -> tuple:
    """
    Calculate the execution accuracy based on the evaluation results of all tasks.

    Args:
        results (List[Dict[str, Any]]): List of results from all tasks.

    Returns:
        tuple: The final execution accuracy,The execution accuracy without revision.
    """
    total_tasks = len(results)
    correct_tasks_r = sum(1 for result in results if result["evaluation"]['revision']['exec_res'])
    correct_tasks_c = sum(1 for result in results if result["evaluation"]['candidate_generation']['exec_res'])
    final_exec_accuracy = correct_tasks_r / total_tasks if total_tasks > 0 else 0
    exec_accuracy_without_revision = correct_tasks_c / total_tasks if total_tasks > 0 else 0
    return final_exec_accuracy, exec_accuracy_without_revision


def main():
    # Load the tasks from the JSON file
    tasks = load_tasks(data_path)

    # Create a new directory for the run results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getenv("OUTPUT_DIR_PATH"), f"run_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run all tasks and store results
    all_results = []
    for t in tasks:
        task = Task(t)
        task_id = task.question_id

        print(f"Running task {task_id}...")
        results = run_task(task)
        store_results(results, output_dir, task_id)
        all_results.append(results)
        print("----------------------------------------------\n")

    # Calculate and output the execution accuracy after revision
    accuracy_r, accuracy_c = calculate_accuracy(all_results)
    print(f"Final Execution accuracy: {accuracy_r:.2%}")
    print(f"Execution accuracy without revision: {accuracy_c:.2%}")

    # Store the accuracy in the results folder
    accuracy_path = os.path.join(output_dir, "execution_accuracy.txt")
    with open(accuracy_path, "w") as file:
        file.write(f"Final Execution accuracy: {accuracy_r:.2%}")
        file.write(f"Execution accuracy without revision: {accuracy_c:.2%}")
    print(f"Execution accuracy stored in {accuracy_path}")


if __name__ == "__main__":
    main()
