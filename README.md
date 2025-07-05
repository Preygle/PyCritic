# PyCritic: Python Code Analysis

This project aims to provide comprehensive analysis for Python code, combining traditional static analysis with machine learning and generative AI techniques.

## Core Concepts & Definitions

### AST (Abstract Syntax Tree)
An Abstract Syntax Tree (AST) is a tree representation of the structural and semantic elements of a program's source code. It's a condensed version of a parse tree, focusing on the essential aspects of the code's structure. ASTs are used by compilers, interpreters, and other tools to analyze, manipulate, and generate code.

### Random Forest
It uses multiple decision trees to make predictions, each tree is somewhat different and the actual classification is done by averaging the results. It reduces error and overfitting, which was the main issue in normal decision trees.

### F1 Score
Harmonic Mean of Precision and Recall.

*   **Precision:** Ratio of Actual positives by total positives identified.
    `True Positives / (True Positives + False Positives)`
*   **Recall:** Ratio of identified positives by actual positives.
    `True Positives / (True Positives + False Negatives)`

### Macro F1 Score
Mean of F1 scores.

### Cross-Validation (CV)
Splits training data into multiple parts, repeats training and testing on different parts (different splits) and averages the result. Makes sure your model isn’t just accidentally doing well on one lucky test split. Here, `cv = 3` means the model is trained and tested 3 times.

### Confusion Matrix
Table used to describe the performance of a classification model. Here, using Multiclass Classifier (more than 2) namely (excellent, good, fair, poor).

### Project Notes
*   Scalar fitting is done only on train data as test data is only supposed to be used for testing and not to be touched while training (can cause overfitting).
---

## Important Links


*   **Download Dataset:** [http://files.srl.inf.ethz.ch/data/py150.tar.gz](http://files.srl.inf.ethz.ch/data/py150.tar.gz)
*   **ML Model Link:** [https://huggingface.co/Preygle/PyCritic/blob/main/code_eval_w_150k.joblib](https://huggingface.co/Preygle/PyCritic/blob/main/code_eval_w_150k.joblib)
*   Running model on Kaggle for easier data loading (and not destroying my CPU).
*   **Generative AI Model for Inference:** `gemini-2.0-flash` (free in Google AI Studio).



### To Do

*   Create VS Code extension for this.

---

## Project Setup

### File Structure

```
PyCritic/
├── .git/
├── .venv/
├── src/
│   └── analyzer/
│       ├── __init__.py
│       ├── api_test.py
│       ├── code_check.py
│       ├── code_suggester.py
│       ├── dataset_extractor.py
│       ├── ml_model.py
|       ├──  sample_env.txt
│       ├── ml_static_result.py
│       ├── parse_python.py
│       ├── static_analyzer.py
│       ├── train_dataset.py
│       ├── transformer_eval.py
│       ├── web_ui.py
│       └── code-suggestion-model/
├── dataset-train-kaggle.ipynb
├── README.md
├── requirements.txt
```

### Environment Variables (`.env`)

This project uses environment variables for API keys and project IDs. Create a file named `.env` in the `src/analyzer/` directory based on the `sample_env.txt` provided in the root directory.

1.  Copy `sample_env.txt` to `src/analyzer/.env`:
    ```bash
    cp sample_env.txt src/analyzer/.env
    ```
    *On Windows:*
    ```bash
    copy sample_env.txt src\analyzer\.env
    ```

2.  Open `src/analyzer/.env` and replace the placeholder values with your actual API keys and project IDs:

    ```
    API_KEY=YOUR_IBM_WATSONX_API_KEY
    URL=YOUR_IBM_WATSONX_URL
    PROJECT_ID=YOUR_IBM_WATSONX_PROJECT_ID
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY
    ```

### Virtual Environment Setup

It's highly recommended to use a virtual environment to manage project dependencies.

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

2.  **Activate the virtual environment:**
    *   **On Windows:**
        ```bash
        .venv\Scripts\activate.bat
        ```
    *   **On macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

### Installation

Once your virtual environment is active, install the required Python packages:

```bash
pip install -r requirements.txt
```

### How to Run

The primary way to interact with PyCritic is through its Streamlit web interface.

1.  **Ensure your virtual environment is active.**
2.  **Navigate to the project root directory.**
3.  **Run the Streamlit application:**
    ```bash
    streamlit run src/analyzer/web_ui.py
    ```
    This will open the web UI in your browser.

Alternatively, you can run the `code_suggester.py` directly for command-line analysis (primarily for development/testing):

```bash
python src/analyzer/code_suggester.py
```