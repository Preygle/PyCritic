import os
import ast
from dotenv import load_dotenv
import google.generativeai as genai
import joblib

# Load environment variables
load_dotenv()

def initialize_ai_model():
    """Initialize the Gemini AI model"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

def load_saved_model(model_path="code_eval_w_150k.joblib"):
    """Load the saved ML model from .joblib"""
    try:
        ml_model = joblib.load(model_path)
        print("ML Model loaded successfully!")
        return ml_model
    except Exception as e:
        print(f"Error loading ML model: {e}")
        return None

from ml_static_result import code_to_trained_format



def create_suggestion_prompt(ml_results, code):
    return f"""
You are a Python code review teacher. A machine learning model has analyzed a piece of Python code and produced the following evaluation:

- **Quality:** {ml_results.get('quality', 'N/A')}
- **Naming:** {ml_results.get('naming', 'N/A')}
- **Style:** {ml_results.get('style', 'N/A')}

Here is the code that was analyzed:
```python
{{code}}
```

Based on this evaluation and the provided code, your task is to provide brief, actionable suggestions to improve the code. Focus on the areas the model flagged as fair or poor.

**Naming and Identifier Guidelines:**

1.  **PEP 8 Compliance:**
    *   **Functions and Variables:** Must be `snake_case` (e.g., `my_variable`, `calculate_sum`).
    *   **Classes:** Must be `PascalCase` (e.g., `MyClass`).
    *   **Constants:** Must be `UPPER_SNAKE_CASE` (e.g., `MAX_VALUE`).
2.  **Descriptive Names:**
    *   Analyze the code's logic to ensure variable and function names are descriptive.
    *   For example, if a variable `x` is used to store a sum, suggest renaming it to `total_sum`.
    *   If a loop control variable is generic (e.g., `i` or `j`), suggest a more descriptive name based on what is being iterated over (e.g., `number` instead of `i` in a loop summing numbers).
    *   This applies to all identifiers: functions, variables, classes, helper functions, etc.

Your response must be formatted as a markdown string, with each suggestion clearly delineated. Use headings and bullet points for readability.

For example:
```markdown
### Code Review Suggestions

**Issue Type:** Naming Convention
**Description:** The function name 'Calculate_sum' does not follow the snake_case convention.
**Suggestion:** Rename 'Calculate_sum' to 'calculate_sum'.

---

**Issue Type:** Descriptive Naming
**Description:** The variable 'x' is not descriptive. It stores the sum of numbers.
**Suggestion:** Rename 'x' to 'sum_of_numbers' for clarity.

---

**Issue Type:** Code Complexity
**Description:** The function has a high cyclomatic complexity.
**Suggestion:** Consider refactoring the function to reduce nesting and improve readability.
```

Begin your analysis now:
"""

from train_dataset import extract_features_from_ast, _score_to_category
import time

def analyze_code_with_ai_and_ml(file_path, ai_model, ml_model, max_retries=3):
    """Analyze code from a file using both ML and GenAI models."""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
    except FileNotFoundError:
        return "Error: The specified file was not found."

    # Get ML analysis
    ast_json = code_to_trained_format(code)
    features = extract_features_from_ast(ast_json)
    
    quality_classifier = ml_model['quality_classifier']
    naming_classifier = ml_model['naming_classifier']
    style_classifier = ml_model['style_classifier']
    scaler = ml_model['scaler']

    features_scaled = scaler.transform(features.reshape(1, -1))

    quality_pred = quality_classifier.predict(features_scaled)[0]
    naming_pred = naming_classifier.predict(features_scaled)[0]
    style_pred = style_classifier.predict(features_scaled)[0]
    
    ml_results = {
        "quality": quality_pred,
        "naming": naming_pred,
        "style": style_pred
    }

    # Get GenAI suggestions with retry logic
    prompt = create_suggestion_prompt(ml_results, code)
    for attempt in range(max_retries):
        try:
            response = ai_model.generate_content(prompt)
            print(f"--- Raw API Response (Attempt {{attempt + 1}}) ---")
            print(response.text)
            print("----------------------------------------")

            ai_suggestions = response.text
            break  # Success, exit the loop

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                ai_suggestions = "Failed to get AI suggestions after multiple attempts."

    return {{
        "ml_analysis": ml_results,
        "ai_suggestions": ai_suggestions
    }}



if __name__ == "__main__":
    # --- IMPORTANT ---
    # Before running, make sure you have a .env file in this directory with your
    # GEMINI_API_KEY.
    #
    # You also need the `code_eval_w_150k.joblib` file in this directory.
    # -----------------
    try:
        print("Initializing AI model...")
        ai_model = initialize_ai_model()

        print("Loading ML model...")
        model_path = "code_eval_w_150k.joblib"
        ml_model = load_saved_model(model_path)

        if ml_model:
            print("\n--- Analyzing code_check.py ---\n")
            # Construct the path to the code_check.py file
            code_check_path = "code_check.py"
            analysis_results = analyze_code_with_ai_and_ml(code_check_path, ai_model, ml_model)

            print("\n=== GenAI Suggestions ===")            
            print(analysis_results["ai_suggestions"])

    except Exception as e:
        print(f"An error occurred: {e}")
