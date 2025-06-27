import os
import ast
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

from ml_static_result import code_to_trained_format

# Load environment variables
load_dotenv()

def initialize_ai_model():
    """Initialize the Watson AI model"""
    apikey = os.getenv("API_KEY")
    url = os.getenv("URL")
    project_id = os.getenv("PROJECT_ID")

    if not all([apikey, url, project_id]):
        raise ValueError("Missing required environment variables")

    model = ModelInference(
        model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
        credentials={"apikey": apikey, "url": url},
        # repetition_penalty=1.2 for prompt to not regenerate same suggestions
        params={
            GenParams.MAX_NEW_TOKENS: 500,
            GenParams.TEMPERATURE: 0.7,
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
            GenParams.REPETITION_PENALTY: 1.2,
        },
        project_id=project_id
    )
    return model


def create_ast_analysis_prompt(ast_data):
    return f"""
You are a Python code review teacher. Analyze the following code's AST structure and provide specific feedback. Your feedback should not inlcude logical, syntax or any other form of error

AST Data: {ast_data}

Your task is to examine this code and provide detailed feedback as a teacher would:

Look through the AST for these specific elements and check each one, if there are none, no need to write anything about it in your re4sponse only write whats wrong
:
- Functions (should be snake_case)
- Classes (should be PascalCase) 
- Variables (should be snake_case)
- Constants (should be UPPER_SNAKE_CASE)
- Imports (make sure the imported library is used and also make sure the chosed alias for the library is correct i.e. correct user if he's importing numpy as pd)
- Code complexity issues
- Any other structural concerns

For each issue found, explain:
- What is wrong (tell me the specific naming convention that is violated)
- How to fix it

Make sure step your suggestions are as brief as possible while encompassing all the rules mentioned above.

Your responses should be brief (must identify any wrong naming convention, tell where the mistake is made and make suggestion. each suggestion should be at max 25 words).
 Be specific about what you found in THIS code, not generic examples. Your suggestion should be more point like and not descriptive
 DO NOT create any summary of your generated answers, i just need violations, corrected name/code and nothing more

Begin your analysis now:
"""


def analyze_code_with_ai(code, model):
    """Convert code to AST and analyze with AI"""
    ast_data = code_to_trained_format(code)
    prompt = create_ast_analysis_prompt(ast_data)
    response = model.generate(prompt=prompt)
    return response['results'][0]['generated_text'] if response and 'results' in response else "Analysis failed"


if __name__ == "__main__":
    # Example Python code to analyze
    test_code = """
CONSTANT_vALUE = 42
def Calculate_sum(a, b):
    return a + b

import pandas as np
class Calculator:
    def __init__(self):
        self.value = 0
    
    def ADD(self, x):
        self.value += x
        self.value += CONSTANT_vALUE
        return self.value
"""

    try:
        print("Initializing AI model...")
        ai_model = initialize_ai_model()

        print("\nAnalyzing code...")
        analysis = analyze_code_with_ai(test_code, ai_model)

        print("\n=== Code Analysis ===")
        print(analysis)
    except Exception as e:
        print(f"Error: {e}")
