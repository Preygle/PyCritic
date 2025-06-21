import ast
import joblib
import os
import numpy as np
from collections import defaultdict
import re
from static_analyzer import StaticCodeAnalyzer

code = """
import numpy as np

class ProperClass:
    def correct_method(self):
        proper_variable = 42
        MAX_vALUE = 100
        _private_var = "secret"
        return proper_variable * MAX_VALUE
"""


def load_saved_model():
    """Load the saved ML model from .joblib"""
    model_file = "code_eval_w_150k.joblib"
    try:
        ml_model = joblib.load(model_file)
        print("Model loaded successfully!")
        return ml_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def code_to_trained_format(code_string):
    nodes = []

    def add_node(node):
        index = len(nodes)
        node_dict = {"type": node.__class__.__name__}

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node_dict["value"] = node.value
        elif isinstance(node, ast.Name):
            ctx = node.ctx.__class__.__name__
            node_dict["type"] = f"Name{ctx}"
            node_dict["value"] = node.id
        elif isinstance(node, ast.alias):
            node_dict["value"] = node.name
        elif isinstance(node, ast.Constant):
            node_dict["value"] = str(node.value)
        elif hasattr(node, 'name'):
            node_dict["value"] = node.name
        elif hasattr(node, 'arg'):
            node_dict["value"] = node.arg

        nodes.append(node_dict)

        children_indices = []
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        child_index = add_node(item)
                        children_indices.append(child_index)
            elif isinstance(value, ast.AST):
                child_index = add_node(value)
                children_indices.append(child_index)

        if children_indices:
            nodes[index]["children"] = children_indices

        return index

    try:
        tree = ast.parse(code_string)
        add_node(tree)
        return nodes
    except Exception as e:
        return [{"type": "Error", "value": str(e)}]


def code_to_ast(code):
    return ast.parse(code)


def static_analyzer(code):
    analyzer = StaticCodeAnalyzer()
    return analyzer.check_naming_conventions(code)


def static_results_to_string(code):
    analyzer = StaticCodeAnalyzer()
    results =  analyzer.check_naming_conventions(code)
    output = ""
    for violation in results['violations']:
        output += f"Line {violation['line']}: {violation['explanation']}\n"
        output += f" Original: {violation['original']}\n"
        output += f" Corrected: {violation['corrected'] if violation['corrected'] else 'No correction needed'}\n"

    output += f"\nTotal violations found: {len(results['violations'])}"
    return output


def analyze_code(code):
    print("=== STATIC ANALYSIS ===")
    static_result = static_results_to_string(code)
    print(static_result)

    print("\n=== ML ANALYSIS ===")
    ast_json = code_to_trained_format(code)
    print(f"Converted to AST format:\n {ast_json}\n\n")

    model_data = load_saved_model()  # This returns the dictionary
    if model_data is not None:
        try:
            # Import the predict function from your training script
            from train_dataset import predict, load_model

            # Load the model using the function from training script
            load_model("code_eval_w_150k.joblib")

            # Now use the predict function
            ml_results = predict(ast_json)

            print("ML Analysis Results:")
            print(f" Quality: {ml_results.get('quality', 'N/A')}")
            print(f" Naming: {ml_results.get('naming', 'N/A')}")
            print(f" Style: {ml_results.get('style', 'N/A')}")

            if 'suggestions' in ml_results:
                print("\nML Suggestions:")
                for suggestion in ml_results['suggestions']:
                    print(f" - {suggestion}")
            return static_result, ml_results
        except Exception as e:
            print(f"Error running ML analysis: {e}")
            print("This might be due to model format incompatibility")
            return ""
    else:
        print("Could not load ML model")

    return None

def runner():
    """Run the analysis on the provided code."""
    analyze_code(code)

# Run the analysis
if __name__ == "__main__":
   analyze_code(code)
    # You can replace 'code' with any other code string you want to analyze
    # For example, you can read from a file or take input from the user
    # with open('your_code_file.py', 'r') as f:
    #     code = f.read()
    #     analyze_code(code)
