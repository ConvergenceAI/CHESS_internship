from typing import Any, List
from src.llm.llm_interface import UnifiedLLMInterface
from dotenv import load_dotenv
import os
from utils.prompt import load_prompt
from utils.response_parsers import list_parser

load_dotenv()
PROMPT_PATH = os.getenv("PROMPT_ROOT_PATH") + "\\keyword_extraction.txt"


def keyword_extraction(task: Any, model: str) -> List[str]:
    """
     Extracts keywords from the task using an LLM

     Args:
         task(Any): The task object containing the evidence and question.
         model(str): The LLM model used to extract the keywords
     Returns:
           List of keywords extracted from the task
    """

    llm = UnifiedLLMInterface()
    prompt = load_prompt(PROMPT_PATH)
    prompt = prompt.format(QUESTION=task.question, HINT=task.evidence)
    response = llm.generate(model, prompt)
    # Parses the output to extract Python list content from markdown
    keywords = list_parser(response)
    return keywords
