import openai

class PromptCompletion:
    def __init__(self, 
                 model="gpt-3.5-turbo", 
                 system_content="""You will act as a Question Answering model. First, just answer the question. Then, provide the elaboration in separate lines. If there has no elaboration statement please return nothing.
Format your response as follows:
Elaboration: [Your elaboration statement]
Answer: [Your answer]

Example #1:
Q: Yes or no: Would a pear sink in water?
A(Part of this system answer): The density of a pear is about 0.6 g/cm^3, which is less than water. Thus, a pear would float. So the answer is no.

Elaboration: The density of a pear is about 0.6 g/cm^3, which is less than water. Thus, a pear would float.
Answer: So the answer is no.

Example #2:
Q: How many keystrokes are needed to type the numbers from 1 to 500? Answer Choices: (a) 1156 (b) 1392 (c) 1480 (d) 1562 (e) 1788
A(Part of this system answer): The answer is B.

Elaboration(Return empty): ''
Answer: The answer is B.""", 
                 temperature=0, 
                 top_p=0, 
                 max_tokens=100):
        """
        Initialize the PromptCompletion class with default parameters.

        Parameters:
        model (str): The OpenAI model to use for generating responses.
        system_content (str): The system message that sets the context and format for the completions.
        temperature (float): Sampling temperature for the model.
        top_p (float): Nucleus sampling parameter.
        max_tokens (int): Maximum number of tokens in the response.
        """
        self.model = model
        self.system_content = system_content
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def get_detailed_completion(self, prompt: str):
        """
        Generate a detailed completion with both an elaboration and an answer.

        Parameters:
        prompt (str): The prompt to send to the OpenAI API.

        Returns:
        dict: A dictionary containing the elaboration and answer.
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )

        response_content = response.choices[0].message.content.strip()

        elaboration = ""
        answer = ""

        # Parse the response to extract elaboration and answer
        for line in response_content.split("\n"):
            if line.startswith("Elaboration:"):
                elaboration = line.replace("Elaboration:", "").strip()
            elif line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()

        result = {
            "elaboration": elaboration,
            "answer": answer if answer else elaboration
        }

        return result

    def get_simple_completion(self, prompt: str):
        """
        Generate a simple completion that only answers the question without any elaboration.

        Parameters:
        prompt (str): The prompt to send to the OpenAI API.

        Returns:
        str: The answer as a string.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You will act as a Question Answering model. Just answer the question."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=0
        )
        return response.choices[0].message.content.strip()
