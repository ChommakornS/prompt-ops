import openai
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model
similarity_model = SentenceTransformer("all-mpnet-base-v2")

def evaluate_response(text1, text2, model):
    """
    Evaluate the response using the SentenceTransformer model.
    
    Parameters:
    text1 (str): The first text to compare.
    text2 (str): The second text to compare.
    model (SentenceTransformer): The model to use for generating embeddings.
    
    Returns:
    float: The similarity score between the two texts.
    """
    emb_a = model.encode([text1])
    emb_b = model.encode([text2])
    similarities = model.similarity(emb_a, emb_b)
    return similarities.item()

def get_completion(prompt: str, model_name: str, system_message: str):
    """
    Get the completion from the OpenAI API.
    
    Parameters:
    prompt (str): The user prompt.
    model_name (str): The name of the OpenAI model to use.
    system_message (str): The system message providing context for the model.
    
    Returns:
    str: The generated response from the model.
    """
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        top_p=0
    )
    return response.choices[0].message.content.strip()

class Test:
    """
    A class representing a test case generation.
    """
    def __init__(self, name, prompt, expected_result, description=None,
                 perturb_method=None, perturb_text=None, capability=None,
                 pass_condition="increase"):
        """
        Initialize a new Test instance.
        
        Parameters:
        name (str): The name of the test.
        prompt (str): The prompt to generate the response.
        expected_result (str): The expected result of the test.
        description (str, optional): A description of the test.
        perturb_method (callable, optional): A method to perturb the prompt.
        perturb_text (str, optional): The perturbed text.
        capability (str, optional): The capability being tested.
        pass_condition (str, optional): The condition to pass the test ('increase' or 'decrease').
        """
        self.name = name
        self.description = description
        self.prompt = prompt
        self.expected_result = expected_result
        self.perturb_method = perturb_method
        self.perturb_text = perturb_text
        self.capability = capability
        self.pass_condition = pass_condition
        self.original_response = None
        self.perturb_response = None
        self.score_original = None
        self.score_perturb = None
        self.model_name = None  

    def run(self, qa_model, model_name, system_message):
        """
        Run the test case by generating and evaluating the responses.
        
        Parameters:
        qa_model (str): The QA model to use.
        model_name (str): The name of the model.
        system_message (str): The system message providing context for the model.
        """
        self.model_name = model_name  
        self.original_response = self.get_response(qa_model, self.prompt, model_name, system_message)
        if self.perturb_method:
            self.perturb_text = self.perturb_method(self.prompt)
        self.perturb_response = self.get_response(qa_model, self.perturb_text, model_name, system_message)

        if self.original_response:
            self.score_original = self.evaluate(similarity_model, self.original_response)
        if self.perturb_response:
            self.score_perturb = self.evaluate(similarity_model, self.perturb_response)

    def get_response(self, qa_model, text, model_name, system_message):
        """
        Get the response from the OpenAI API or another model.
        
        Parameters:
        qa_model (str): The QA model to use.
        text (str): The text to get a response for.
        model_name (str): The name of the model.
        system_message (str): The system message providing context for the model.
        
        Returns:
        str: The generated response.
        """
        if not text:
            return None

        if qa_model == "openai":
            result = get_completion(text, model_name, system_message)
        else:
            result = qa_model(text)

        if isinstance(result, list) and result:
            return result[0]['label']
        return result

    def evaluate(self, model, response):
        """
        Evaluate the response using the specified model.
        
        Parameters:
        model (SentenceTransformer): The model to use for evaluation.
        response (str): The response to evaluate.
        
        Returns:
        float: The similarity score between the response and the expected result.
        """
        if response is None:
            return None
        return evaluate_response(response, self.expected_result, model)

    def summarize(self):
        """
        Summarize the test case and return all results.
        
        Returns:
        dict: A dictionary summarizing the test case results.
        """
        fail = False
        if self.score_original is not None and self.score_perturb is not None:
            if self.pass_condition == "decrease":
                if self.score_perturb >= self.score_original:
                    fail = True
            elif self.pass_condition == "increase":
                if self.score_perturb < self.score_original:
                    fail = True

        return {
            'name': self.name,
            'description': self.description,
            'prompt': self.prompt,
            'expected_result': self.expected_result,
            'perturb_text': self.perturb_text,
            'pass_condition': self.pass_condition,
            'capability': self.capability,
            'response_original': self.original_response,
            'response_perturb': self.perturb_response,
            'score_original': self.score_original,
            'score_perturb': self.score_perturb,
            'fail': fail,
            'model_name': self.model_name  
        }
