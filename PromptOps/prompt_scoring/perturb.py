import random

def perturb(text):
    """
    Swap one random character with its neighboring character in the text.
    
    Parameters:
    text (str): The input text to be perturbed.

    Returns:
    str: The perturbed text.
    """
    if len(text) < 2:
        return text

    index = random.randint(0, len(text) - 2)
    if text[index].isspace() or text[index + 1].isspace():
        return text  

    perturbed_text = (
        text[:index] +
        text[index + 1] +
        text[index] +
        text[index + 2:]
    )

    return perturbed_text
