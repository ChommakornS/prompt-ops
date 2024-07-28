from setuptools import setup, find_packages

setup(
    name='PromptOps',
    version='0.1.0',
    description='A Python library for prompt scoring and suggestion',
    author='ChommakornS',
    author_email='chommakorn.son@gmail.com',
    packages=find_packages(),
    install_requires=[
        'openai==0.28',
        'pandas',
        'sentence-transformers',
        'scikit-learn',
        'langchain'
    ],
)
