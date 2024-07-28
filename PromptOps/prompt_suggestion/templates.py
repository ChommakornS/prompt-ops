from langchain.prompts import PromptTemplate

class Template:
    def __init__(self, default_prefix, example_template, examples, suffix, example_separator="\n\n", instruction=""):
        """
        Initialize the Template class with default values.

        Parameters:
        default_prefix (str): Default prefix for the prompt.
        example_template (str): Template for examples.
        examples (list): List of examples to be used in the prompt.
        suffix (str): Suffix to be added after the examples.
        example_separator (str): Separator between examples. Defaults to "\n\n".
        instruction (str): Additional instructions. Defaults to an empty string.
        """
        self.default_prefix = default_prefix
        self.example_template = example_template
        self.examples = examples
        self.suffix = suffix
        self.example_separator = example_separator
        self.instruction = instruction

    def create_prompt(self, prefix=None, user_input=None, context=""):
        """
        Create a full prompt using the given parameters.

        Parameters:
        prefix (str): Prefix for the prompt. Defaults to None.
        user_input (str): User input to be added to the prompt. Defaults to None.
        context (str): Context to be added to the prompt. Defaults to an empty string.

        Returns:
        str: Full prompt.
        """
        prefix = prefix if prefix else self.default_prefix
        full_prompt = create_full_prompt(
            prefix, self.examples, self.example_template, self.suffix, user_input, self.example_separator, context
        )
        return f"{full_prompt}"

    def get_input_variables(self):
        """
        Get input variables from the example template.

        Returns:
        list: List of input variables.
        """
        return [var.split(":")[0] for var in self.example_template.strip().split("\n")]

def create_full_prompt(prefix, examples, example_prompt, suffix, user_input, example_separator="\n\n", context=""):
    """
    Create a full prompt with examples and user input.

    Parameters:
    prefix (str): Prefix for the prompt.
    examples (list): List of examples.
    example_prompt (str): Template for examples.
    suffix (str): Suffix for the prompt.
    user_input (str): User input to be added to the prompt.
    example_separator (str): Separator between examples. Defaults to "\n\n".
    context (str): Context to be added to the prompt. Defaults to an empty string.

    Returns:
    str: Full prompt.
    """
    example_section = example_separator.join(
        [example_prompt.format(**example) for example in examples]
    )
    return f"{context}\n{prefix}\n{example_section}\n{suffix.format(query=user_input)}"

# StdSent template
def std_sent(examples, example_template, suffix=None, example_separator="\n\n", instruction="", prefix=None):
    """
    Create a standard sentiment classification prompt template.

    Parameters:
    examples (list): List of examples.
    example_template (str): Template for examples.
    suffix (str): Suffix for the prompt. Defaults to None.
    example_separator (str): Separator between examples. Defaults to "\n\n".
    instruction (str): Additional instructions. Defaults to an empty string.
    prefix (str): Prefix for the prompt. Defaults to None.

    Returns:
    Template: Template object.
    """
    default_prefix = """The following sentences provide different sentiments.
Please classify each sentence as negative, positive, or neutral."""
    template = Template(
        default_prefix=prefix or default_prefix,
        example_template=example_template,
        examples=examples,
        suffix=suffix or """
Text: {query}
Sentiment: """,
        example_separator=example_separator,
        instruction=instruction
    )
    return template

# CotSent template
def cot_sent(examples, example_template, suffix=None, example_separator="\n\n", instruction="", prefix=None):
    """
    Create a chain-of-thought sentiment classification prompt template.

    Parameters:
    examples (list): List of examples.
    example_template (str): Template for examples.
    suffix (str): Suffix for the prompt. Defaults to None.
    example_separator (str): Separator between examples. Defaults to "\n\n".
    instruction (str): Additional instructions. Defaults to an empty string.
    prefix (str): Prefix for the prompt. Defaults to None.

    Returns:
    Template: Template object.
    """
    default_prefix = """The following sentences provide different sentiments.
Please classify each sentence as negative, positive, or neutral."""
    template = Template(
        default_prefix=prefix or default_prefix,
        example_template=example_template,
        examples=examples,
        suffix=suffix or """
Text: {query}
Sentiment: Let's Think Step by Step""",
        example_separator=example_separator,
        instruction=instruction
    )
    return template

# StdQna template
def std_qna(examples, example_template, suffix=None, example_separator="\n\n", instruction=None, prefix=None):
    """
    Create a standard question-answering prompt template.

    Parameters:
    examples (list): List of examples.
    example_template (str): Template for examples.
    suffix (str): Suffix for the prompt. Defaults to None.
    example_separator (str): Separator between examples. Defaults to "\n\n".
    instruction (str): Additional instructions. Defaults to None.
    prefix (str): Prefix for the prompt. Defaults to None.

    Returns:
    Template: Template object.
    """
    default_prefix = """Answer the question based on the context below."""
    template = Template(
        default_prefix=prefix or default_prefix,
        example_template=example_template,
        examples=examples,
        suffix=suffix or """
Question: {query}
Answer: """,
        example_separator=example_separator,
        instruction=instruction or ""
    )
    return template

# CotQna template
def cot_qna(examples, example_template, suffix=None, example_separator="\n\n", instruction=None, prefix=None):
    """
    Create a chain-of-thought question-answering prompt template.

    Parameters:
    examples (list): List of examples.
    example_template (str): Template for examples.
    suffix (str): Suffix for the prompt. Defaults to None.
    example_separator (str): Separator between examples. Defaults to "\n\n".
    instruction (str): Additional instructions. Defaults to None.
    prefix (str): Prefix for the prompt. Defaults to None.

    Returns:
    Template: Template object.
    """
    default_prefix = """Answer the question based on the context below."""
    template = Template(
        default_prefix=prefix or default_prefix,
        example_template=example_template,
        examples=examples,
        suffix=suffix or """
Question: {query}
Answer: Let's think step by step.""",
        example_separator=example_separator,
        instruction=instruction or ""
    )
    return template
