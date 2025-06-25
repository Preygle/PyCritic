import os
import ast
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

from ml_static_result import code_to_trained_format

# Load environment variables
load_dotenv()


# def code_to_trained_format(code_string):
#     """Convert Python code to AST format (your original function)"""
#     nodes = []

#     def add_node(node):
#         index = len(nodes)
#         node_dict = {"type": node.__class__.__name__}

#         if isinstance(node, ast.Constant) and isinstance(node.value, str):
#             node_dict["value"] = node.value
#         elif isinstance(node, ast.Name):
#             ctx = node.ctx.__class__.__name__
#             node_dict["type"] = f"Name{ctx}"
#             node_dict["value"] = node.id
#         elif isinstance(node, ast.alias):
#             node_dict["value"] = node.name
#         elif isinstance(node, ast.Constant):
#             node_dict["value"] = str(node.value)
#         elif hasattr(node, 'name'):
#             node_dict["value"] = node.name
#         elif hasattr(node, 'arg'):
#             node_dict["value"] = node.arg

#         nodes.append(node_dict)

#         children_indices = []
#         for field, value in ast.iter_fields(node):
#             if isinstance(value, list):
#                 for item in value:
#                     if isinstance(item, ast.AST):
#                         child_index = add_node(item)
#                         children_indices.append(child_index)
#             elif isinstance(value, ast.AST):
#                 child_index = add_node(value)
#                 children_indices.append(child_index)

#         if children_indices:
#             nodes[index]["children"] = children_indices

#         return index

#     try:
#         tree = ast.parse(code_string)
#         add_node(tree)
#         return nodes
#     except Exception as e:
#         return [{"type": "Error", "value": str(e)}]


def initialize_ai_model():
    """Initialize the Watson AI model"""
    apikey = os.getenv("API_KEY")
    url = os.getenv("URL")
    project_id = os.getenv("PROJECT_ID")

    if not all([apikey, url, project_id]):
        raise ValueError("Missing required environment variables")

    model = ModelInference(
        model_id="meta-llama/llama-3-3-70b-instruct",
        credentials={"apikey": apikey, "url": url},
        params={
            GenParams.MAX_NEW_TOKENS: 500,
            GenParams.TEMPERATURE: 0.7,
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE
        },
        project_id=project_id
    )
    return model


def create_ast_analysis_prompt(ast_data):
    """Create prompt for AST analysis"""
    return f"""
Analyze this Python AST and provide:
1. Code functionality assessment
2. Quality evaluation
3. Improvement suggestions

AST Data:
{ast_data}

Focus on the AST structure and node patterns.
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
def calculate_sum(a, b):
    return a + b

class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
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
