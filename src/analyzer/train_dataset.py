import json
import numpy as np
from collections import defaultdict
import pickle
import os
import re

import matplotlib.pyplot as plt
from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Directory where Kaggle datasets are typically located after unzipping
DATA_DIR = "/kaggle/working/py150/"

# Filenames for the dataset
# Note: The actual dataset from Kaggle has '.jsonl' extension
TRAIN_FILENAME = "python100k_train.json"
EVAL_FILENAME = "python50k_eval.json"

# Filepath for the saved model in the output directory
MODEL_FILENAME = "code_eval_w_150k.pkl"
MODEL_PATH = f"/kaggle/working/{MODEL_FILENAME}"

# Training parameters
MAX_SAMPLES = 100000
RANDOM_STATE = 42


class MLCodeCriticAST:
    # Initializer Constructor
    def __init__(self):
        # 3 Random Forsest with 50 decision trees
        self.quality_classifier = RandomForestClassifier(
            n_estimators=50, random_state=RANDOM_STATE)
        self.naming_classifier = RandomForestClassifier(
            n_estimators=50, random_state=RANDOM_STATE)
        self.style_classifier = RandomForestClassifier(
            n_estimators=50, random_state=RANDOM_STATE)
        # Normalizing Features
        self.scaler = StandardScaler()
        # Check whether model is trainind or not
        self.is_trained = False

    def extract_features_from_ast(self, ast_json):
        # Store count of various node types (class, func, etc)
        node_counts = defaultdict(int)
        # identifier name list
        names = []
        for node in ast_json:
            if isinstance(node, dict) and 'type' in node:
                node_type = node['type']
                node_counts[node_type] += 1  # node type counter incrementor
                # check for indentifier names
                if 'value' in node and isinstance(node['value'], str):
                    names.append(node['value'])

        # Naming and style features
        # mean lenght of identifiers
        name_len = np.mean([len(n) for n in names]) if names else 0

        # Snake and Camel case check
        snake_case = sum(
            1 for n in names
            if re.fullmatch(r'[a-z]+(_[a-z]+)+', n)
        )

        camel_case = sum(
            1 for n in names
            if re.fullmatch(r'[a-z]+([A-Z][a-z0-9]*)+', n)
        )

        # Complexity
        complexity = node_counts['If'] + \
            node_counts['For'] + node_counts['While']

        # Array of extracted features
        features = np.array([
            node_counts['FunctionDef'], node_counts['ClassDef'], node_counts['Assign'],
            node_counts['Import'] + node_counts['ImportFrom'], complexity,
            node_counts['NameStore'], node_counts['NameLoad'], len(
                names), len(ast_json),
            node_counts['Call'], node_counts['attr'], node_counts['Str'], node_counts['Return'],
            node_counts['arguments'], node_counts['body'],
            name_len, snake_case, camel_case, 0, 0  # Padding for consistent feature count
        ], dtype=float)
        return features

    def load_py150_jsonl(self, filepath, max_samples):
        data = []
        print(f"Loading {max_samples} samples...")
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return []

        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    ast_json = json.loads(line)
                    features = self.extract_features_from_ast(ast_json)

                    # Heuristic-based labeling for training
                    quality = min(100, max(0, 30  # Lower base score
                                        # More points for functions
                                        + 15 * (features[0] > 1)
                                        # More points for classes
                                        + 15 * (features[1] > 0)
                                        # Bonus for good imports
                                        + 10 * (features[3] > 2)
                                        # Return statements
                                        + 10 * (features[12] >= 1)
                                        # Extra bonus for many functions
                                        + 15 * (features[0] > 3)
                                        # Bigger complexity penalty
                                        - 15 * (features[4] > 10)
                                        # Severe complexity penalty
                                        - 20 * (features[4] > 20)
                                        # Penalty for too few names
                                        - 10 * (features[7] < 3)
                                        ))

                    naming = min(100, max(0, 20   # Lower base score
                                        # Longer names bonus
                                        + 20 * (features[15] > 8)
                                        # Good snake_case usage
                                        + 15 * (features[16] >= 3)
                                        # Excellent snake_case
                                        + 10 * (features[16] >= 5)
                                        - 20 * (features[16] == 0 and features[17]
                                                == 0)  # No convention
                                        # Prefer snake_case
                                        - 15 * (features[17] > features[16])
                                        # Too short names
                                        - 10 * (features[15] < 4)
                                        ))

                    style = min(100, max(0, 25    # Lower base score
                                + 15 * (features[3] > 0)    # Has imports
                                + 10 * (features[3] > 2)    # Good import usage
                                + 15 * (features[12] >= 1)  # Has returns
                                + 10 * (features[9] > 5)    # Good function calls
                                - 15 * (features[11] > 8)   # Too many strings
                                - 10 * (features[11] > 15)  # Way too many strings
                                - 15 * (features[8] < 10)   # Too few total nodes
                                ))

                    data.append({
                        "features": features,
                        "quality_score": quality,
                        "naming_score": naming,
                        "style_score": style
                    })
                except json.JSONDecodeError:
                    continue

        print(f"Successfully loaded {len(data)} samples.")
        return data

    def train_models(self, train_data):
        print("\nTraining Model")
        X = np.array([d["features"] for d in train_data])

        # Creates labels for all 3 score types (excellent, good, fair, poor)
        y_quality = np.array([self._score_to_category(
            d["quality_score"]) for d in train_data])
        y_naming = np.array([self._score_to_category(
            d["naming_score"]) for d in train_data])
        y_style = np.array([self._score_to_category(d["style_score"])
                        for d in train_data])

        # Split data for all models
        X_train, X_test, yq_train, yq_test, yn_train, yn_test, ys_train, ys_test = train_test_split(
            X, y_quality, y_naming, y_style, test_size=0.2, random_state=RANDOM_STATE, stratify=y_quality
        )

        # Normalizing data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Tune, train, and evaluate each model
        self.quality_classifier = self._tune_and_train(
            X_train_scaled, yq_train, 'Quality')
        self._evaluate_and_plot(X_test_scaled, yq_test,
                                self.quality_classifier, 'Quality')

        self.naming_classifier = self._tune_and_train(
            X_train_scaled, yn_train, 'Naming')
        self._evaluate_and_plot(X_test_scaled, yn_test,
                                self.naming_classifier, 'Naming')

        self.style_classifier = self._tune_and_train(
            X_train_scaled, ys_train, 'Style')
        self._evaluate_and_plot(X_test_scaled, ys_test,
                                self.style_classifier, 'Style')

        self.is_trained = True
        print("\n=== Training Complete ===")
    
    def _tune_and_train(self, X_train, y_train, model_name):
        print(f"\n--- Tuning {model_name} Model ---")

        # Define random Hyper Parameters for Random Forest
        param_dist = {
            'n_estimators': randint(50, 200), 'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 11), 'min_samples_leaf': randint(1, 5)
        }

        # Using 15 x 3 = 45 () Decision trees for random forest
        rand_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE), param_distributions=param_dist,
            n_iter=15, cv=3, scoring='f1_macro', random_state=RANDOM_STATE, n_jobs=-1
        )

        # Tuning process
        rand_search.fit(X_train, y_train)

        print(f"Best Score: {rand_search.best_score_:.3f}")
        return rand_search.best_estimator_

# Evaluating on train data and displaying Confusion Matrix


    def _evaluate_and_plot(self, X_test, y_test, model, model_name):
        print(f"\n==Evaluating {model_name} Model==")
        # Predict based on test data
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, cmap=plt.cm.Blues)
        plt.title(f'{model_name} Model Confusion Matrix')
        plt.show()

    # Assign score to categories


    def _score_to_category(self, score):
        if score >= 65:
            return 'excellent'
        if score >= 55:
            return 'good'
        if score >= 45:
            return 'fair'
        return 'poor'


critic = MLCodeCriticAST()
train_filepath = os.path.join(DATA_DIR, TRAIN_FILENAME)
train_data = critic.load_py150_jsonl(train_filepath, max_samples=MAX_SAMPLES)

if train_data:
    critic.train_models(train_data)
else:
    print("Skipping training because no data was loaded. Check file paths and names.")

if critic.is_trained:
    critic.save(MODEL_PATH)
else:
    print("Model not trained. Skipping save.")

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    inference_critic = MLCodeCriticAST.load(MODEL_PATH)
    eval_size = 50000
    eval_filepath = os.path.join(DATA_DIR, EVAL_FILENAME)
    if os.path.exists(eval_filepath):
        print(f"\n==Critiquing {eval_size} samples from the evaluation set:==")
        with open(eval_filepath, "r") as f:
            for i, line in enumerate(f):
                if i >= eval_size:
                    break
                try:
                    # Read each code sample
                    ast_json = json.loads(line)

                    # Store it in result
                    result = inference_critic.predict(ast_json)

                    # Print category and suggestion
                    print(f"\n--- Sample {i+1} ---")
                    print(f"  Quality: {result['quality']}")
                    print(f"  Naming:  {result['naming']}")
                    print(f"  Style:   {result['style']}")
                    print("  Suggestions:")
                    for suggestion in result['suggestions']:
                        print(f"    - {suggestion}")
                except json.JSONDecodeError:
                    continue
    else:
        print(
            f"Evaluation file not found at {eval_filepath}. Cannot run inference.")
else:
    print("Saved model not found. Please run the training and saving cells first.")
