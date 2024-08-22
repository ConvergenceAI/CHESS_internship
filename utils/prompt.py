def load_prompt(file_path: str) -> str:
    """
    Load prompt template from a text file.

    Args:
        file_path (str): The path to the text file containing the prompt template.

    Returns:
        str: The prompt template as a string.
    """
    with open(file_path, 'r') as file:
        prompt_template = file.read()
    return prompt_template
