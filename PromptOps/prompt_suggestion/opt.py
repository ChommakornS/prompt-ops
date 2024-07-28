import openai

def get_standard_suggestion(prompt, expected_result, cosine_score, model="gpt-3.5-turbo", temperature=0, top_p=0, max_tokens=100):
    """
    Generates an improved standard prompt based on the current prompt, expected result, and cosine similarity score.

    Parameters:
    prompt (str): The current prompt that needs improvement.
    expected_result (str): The expected response for the prompt.
    cosine_score (float): The cosine similarity score between the actual response and the expected response.
    model (str): The OpenAI model to use for generating the suggestion. Default is "gpt-3.5-turbo".
    temperature (float): Sampling temperature. Default is 0.
    top_p (float): Nucleus sampling parameter. Default is 0.
    max_tokens (int): Maximum number of tokens in the response. Default is 100.

    Returns:
    str: The improved prompt.
    """
    system_prompt = (
        f"The current prompt is: '{prompt}'. The response to this prompt was not as expected. "
        f"The expected response is: '{expected_result}'. The cosine similarity score between the response and the expected result is: {cosine_score:.2f}.\n\n"
        f"Improve the current prompt to better match the expected result: '{expected_result}'. "
        f"If there is no shot example in the current prompt, you can modify the instruction to make the output style and response closer to the expected result, but you cannot modify the last shot and cannot delete the label of the shot and you cannot add a new shot. "
        f"If there are many shot examples in the current prompt, you can modify the instruction and the second line of each shot to make the output style and response closer to the expected result, but you cannot modify the last shot and cannot delete the label of the second line of the shot and you cannot add a new shot. "
        f"If there is one shot example in the current prompt, you can modify the instruction and the second line of the first shot to make the output style and response closer to the expected result, but you cannot modify the last shot and cannot delete the label of the second line of the shot and you cannot add a new shot. "
        f"Please provide only the improved prompt. You need to think about what the meaning of {expected_result} is and make the new prompt generate an answer that matches the expected answer."
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that helps improve prompt sentences. You are prohibited from saying anything else. You can only provide a suggested prompt. You can only modify.\n\n"},
            {"role": "user", "content": system_prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def get_cot_suggestion(prompt, expected_result, cosine_score, model="gpt-3.5-turbo", temperature=0, top_p=0, max_tokens=100):
    """
    Generates an improved chain-of-thought (CoT) prompt based on the current prompt, expected result, and cosine similarity score.

    Parameters:
    prompt (str): The current prompt that needs improvement.
    expected_result (str): The expected response for the prompt.
    cosine_score (float): The cosine similarity score between the actual response and the expected response.
    model (str): The OpenAI model to use for generating the suggestion. Default is "gpt-3.5-turbo".
    temperature (float): Sampling temperature. Default is 0.
    top_p (float): Nucleus sampling parameter. Default is 0.
    max_tokens (int): Maximum number of tokens in the response. Default is 100.

    Returns:
    str: The improved prompt.
    """
    system_prompt = (
        f"The current prompt is: '{prompt}'. The response to this prompt did not meet expectations. "
        f"The expected response is: '{expected_result}'. The cosine similarity score between the response and the expected result is: {cosine_score:.2f}.\n\n"
        f"Improve the current prompt to better match the expected result: '{expected_result}'. "
        "Please suggest an improved prompt. Don't change or delete the label. You cannot modify the last line."
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": 
             "You are an assistant that helps improve prompt sentences. You are prohibited from saying anything else. You can only provide a suggested prompt.\n\n"
             "If there is no shot example, you need to improve the first line by adding more detail. Don't delete the label and don't modify the last shot.\n"
             "If there is one shot example, you need to improve the second line of the first shot by providing the thinking of each step, except the second line in the last shot, you cannot modify 'A:' in the last shot. Don't delete the label or modify 'A:' in the last line.\n"
             "If there are many shot examples, you need to improve the second line of each shot by providing the thinking of each step, except the second line in the last shot, you cannot modify. Don't delete the label.\n"
             "Don't change or delete the label. You cannot modify the last line. For sentiment analysis prompts, you need to identify in the instruction to classify into negative, positive, and neutral.\n"
             "Provide only the improved prompt, don't say 'the improved prompt is/could be.'"
            },
            {"role": "user", "content": system_prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()
