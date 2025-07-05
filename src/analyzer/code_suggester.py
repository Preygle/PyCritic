import ast
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai
import joblib
from ml_static_result import code_to_trained_format
from static_analyzer import StaticCodeAnalyzer
from train_dataset import extract_features_from_ast

load_dotenv()

class CodeSuggester:
    def __init__(self):
        self.ai_model = self._initialize_ai_model()
        self.ml_model = self._load_saved_model()

    def _initialize_ai_model(self):
        """Initialize the Gemini AI model"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model

    def _load_saved_model(self, model_path="code_eval_w_150k.joblib"):
        """Load the saved ML model from .joblib"""
        try:
            ml_model = joblib.load(model_path)
            print("ML Model loaded successfully!")
            return ml_model
        except Exception as e:
            print(f"Error loading ML model: {e}")
            return None

    def generate_suggestions(self, code_string: str) -> dict:
        # 1. Static Analysis for Naming Conventions
        static_analyzer = StaticCodeAnalyzer()
        naming_violations = static_analyzer.check_naming_conventions(code_string)

        # 2. ML-based Code Scoring
        ml_results = self._get_ml_analysis(code_string)

        # 3. GenAI for Import Analysis and Summary
        import_suggestions, summary = self._get_ai_suggestions(code_string)

        return {
            "naming_violations": naming_violations,
            "ml_results": ml_results,
            "import_suggestions": import_suggestions,
            "summary": summary
        }

    def _get_ml_analysis(self, code_string: str) -> dict:
        if not self.ml_model:
            return {}

        ast_json = code_to_trained_format(code_string)
        features = extract_features_from_ast(ast_json)
        
        quality_classifier = self.ml_model['quality_classifier']
        naming_classifier = self.ml_model['naming_classifier']
        style_classifier = self.ml_model['style_classifier']
        scaler = self.ml_model['scaler']

        features_scaled = scaler.transform(features.reshape(1, -1))

        quality_pred = quality_classifier.predict(features_scaled)[0]
        naming_pred = naming_classifier.predict(features_scaled)[0]
        style_pred = style_classifier.predict(features_scaled)[0]
        
        return {
            "quality": quality_pred,
            "naming": naming_pred,
            "style": style_pred
        }

    def _get_ai_suggestions(self, code_string: str) -> tuple[str, str]:
        import_prompt = self._create_import_prompt(code_string)
        summary_prompt = self._create_summary_prompt(code_string)
        try:
            import_response = self.ai_model.generate_content(import_prompt)
            summary_response = self.ai_model.generate_content(summary_prompt)
            return import_response.text, summary_response.text
        except Exception as e:
            return f"Error generating AI suggestions: {e}", ""

    def _create_import_prompt(self, code: str) -> str:
        return f"""
        You are a Python code analysis tool. Your ONLY task is to analyze the import statements in the following code and provide feedback in a specific format. Do NOT generate any other content.

        **Code:**
        ```python
        {code}
        ```

        **Your ONLY Task:**

        1.  **Identify Unused Imports:** Find any imported modules or names that are not used in the code.
        2.  **Check for Incorrect Aliases:** Look for unconventional import aliases, such as `import pandas as np` (should be `pd`) or `import numpy as pd` (should be `np`).

        **Output Format (Strict):**

        - Use markdown formatting.
        - Use the exact headings "Unused Imports" and "Incorrect Import Aliases".
        - For each suggestion, include the line number and the suggested change, formatted as a markdown list item (e.g., `- **Line X:** Suggestion`).
        - Do NOT add any other text, explanations, or summaries.

        **Example:**

        **Unused Imports**
        - **Line 2:** Remove unused import 'os'.

        **Incorrect Import Aliases**
        - **Line 1:** The conventional alias for 'pandas' is 'pd', not 'np'. Consider changing to 'import pandas as pd'.

        Begin your analysis now. Remember, ONLY import analysis.
        """

    def _create_summary_prompt(self, code: str) -> str:
        return f"""
        You are a Python code analysis tool. Your ONLY task is to provide a brief, high-level summary of the following code quality feedback in 5 lines or less. Do NOT generate any other content.

        **Analysis:**
        {self._get_ml_analysis(code)}

        **Your ONLY Task:**
        Provide a short, high-level summary of the code quality issues and suggest general improvements. Each suggestion should be a markdown list item (e.g., `- Suggestion text`). Do NOT include any code snippets or specific examples.
        """

    def _format_markdown_output(self, naming_violations, ml_results, import_suggestions, summary) -> str:
        markdown_output = "### Code Review Suggestions\n\n"

        # Naming Violations
        if naming_violations.get('violations'):
            markdown_output += "**Naming Conventions**\n"
            for violation in naming_violations['violations']:
                markdown_output += f"- **Line {violation['line']}:** `{violation['original']}`\n"
                markdown_output += f"  - **Suggestion:** {violation['explanation']}\n"
                if violation.get('corrected'):
                    markdown_output += f"  - **Correction:** `{violation['corrected']}`\n"
            markdown_output += "---\n"

        # Import Suggestions
        if import_suggestions:
            markdown_output += "**Import Analysis**\n"
            markdown_output += f"{import_suggestions}\n"
            markdown_output += "---\n"

        # ML Code Scoring
        if ml_results:
            markdown_output += "**Code Quality Score**\n"
            markdown_output += f"- **Quality:** {ml_results.get('quality', 'N/A')}\n"
            markdown_output += f"- **Naming:** {ml_results.get('naming', 'N/A')}\n"
            markdown_output += f"- **Style:** {ml_results.get('style', 'N/A')}\n"
            markdown_output += "---\n"

        # AI-Generated Summary
        if summary:
            markdown_output += "### Summary of Suggestions\n"
            markdown_output += f"{summary}\n"

        return markdown_output

if __name__ == '__main__':
    suggester = CodeSuggester()
    test_code = """
import pandas as np
import os
from sys import exit

CONSTANT_vALUE = 42

class my_class:
    def BAD_FUNCTION_NAME(self, some_val):
        local_var = some_val + CONSTANT_vALUE
        return local_var

def anotherFunction():
    pass
"""
    markdown_output = suggester.generate_suggestions(test_code)
    print(markdown_output)