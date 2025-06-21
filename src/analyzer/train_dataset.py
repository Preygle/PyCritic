import json
import numpy as np
from collections import defaultdict
import pickle
import os
import re
import joblib

import matplotlib.pyplot as plt
from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Directory where Kaggle datasets are typically located after unzipping
DATA_DIR = ""

# Filenames for the dataset
# Note: The actual dataset from Kaggle has '.jsonl' extension
TRAIN_FILENAME = "python100k_train.json"
EVAL_FILENAME = "python50k_eval.json"

# Filepath for the saved model in the output directory
MODEL_FILENAME = "code_eval_w_150k.joblib"
MODEL_PATH = f"{MODEL_FILENAME}"

# Training parameters
MAX_SAMPLES = 100000
RANDOM_STATE = 42

# Global model variables
quality_classifier = RandomForestClassifier(
    n_estimators=50, random_state=RANDOM_STATE)
naming_classifier = RandomForestClassifier(
    n_estimators=50, random_state=RANDOM_STATE)
style_classifier = RandomForestClassifier(
    n_estimators=50, random_state=RANDOM_STATE)
scaler = StandardScaler()
is_trained = False


def extract_features_from_ast(ast_json):
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


def load_py150_jsonl(filepath, max_samples):
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
                features = extract_features_from_ast(ast_json)

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


def _tune_and_train(X_train, y_train, model_name):
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


def _evaluate_and_plot(X_test, y_test, model, model_name):
    print(f"\n==Evaluating {model_name} Model==")
    # Predict based on test data
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, cmap=plt.cm.Blues)
    plt.title(f'{model_name} Model Confusion Matrix')
    plt.show()


def _score_to_category(score):
    if score >= 65:
        return 'excellent'
    if score >= 55:
        return 'good'
    if score >= 45:
        return 'fair'
    return 'poor'


def train_models(train_data):
    global quality_classifier, naming_classifier, style_classifier, scaler, is_trained

    print("\nTraining Model")
    X = np.array([d["features"] for d in train_data])

    # Creates labels for all 3 score types (excellent, good, fair, poor)
    y_quality = np.array([_score_to_category(d["quality_score"])
                         for d in train_data])
    y_naming = np.array([_score_to_category(d["naming_score"])
                        for d in train_data])
    y_style = np.array([_score_to_category(d["style_score"])
                       for d in train_data])

    # Split data for all models
    X_train, X_test, yq_train, yq_test, yn_train, yn_test, ys_train, ys_test = train_test_split(
        X, y_quality, y_naming, y_style, test_size=0.2, random_state=RANDOM_STATE, stratify=y_quality
    )

    # Normalizing data
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Tune, train, and evaluate each model
    quality_classifier = _tune_and_train(X_train_scaled, yq_train, 'Quality')
    _evaluate_and_plot(X_test_scaled, yq_test, quality_classifier, 'Quality')

    naming_classifier = _tune_and_train(X_train_scaled, yn_train, 'Naming')
    _evaluate_and_plot(X_test_scaled, yn_test, naming_classifier, 'Naming')

    style_classifier = _tune_and_train(X_train_scaled, ys_train, 'Style')
    _evaluate_and_plot(X_test_scaled, ys_test, style_classifier, 'Style')

    is_trained = True
    print("\n=== Training Complete ===")


def save_model(filepath):
    global quality_classifier, naming_classifier, style_classifier, scaler, is_trained

    if not is_trained:
        print("Model not trained yet!")
        return

    model_data = {
        'quality_classifier': quality_classifier,
        'naming_classifier': naming_classifier,
        'style_classifier': style_classifier,
        'scaler': scaler,
        'is_trained': is_trained
    }

    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    global quality_classifier, naming_classifier, style_classifier, scaler, is_trained

    model_data = joblib.load(filepath)

    quality_classifier = model_data['quality_classifier']
    naming_classifier = model_data['naming_classifier']
    style_classifier = model_data['style_classifier']
    scaler = model_data['scaler']
    is_trained = model_data['is_trained']

    print(f"Model loaded from {filepath}")


def _generate_suggestions(quality, naming, style, features):
    suggestions = []

    if quality in ['poor', 'fair']:
        if features[0] < 2:  # Few functions
            suggestions.append(
                "Consider breaking code into more functions for better modularity")
        if features[4] > 10:  # High complexity
            suggestions.append(
                "Reduce cyclomatic complexity by simplifying conditional logic")
        if features[12] < 1:  # No return statements
            suggestions.append(
                "Add return statements to functions for clarity")

    if naming in ['poor', 'fair']:
        if features[15] < 4:  # Short names
            suggestions.append(
                "Use more descriptive variable and function names")
        if features[16] == 0 and features[17] == 0:  # No naming convention
            suggestions.append(
                "Follow consistent naming conventions (prefer snake_case)")
        if features[17] > features[16]:  # More camelCase than snake_case
            suggestions.append(
                "Consider using snake_case instead of camelCase for Python")

    if style in ['poor', 'fair']:
        if features[3] == 0:  # No imports
            suggestions.append("Organize imports at the top of the file")
        if features[11] > 8:  # Too many string literals
            suggestions.append(
                "Consider using constants for repeated string values")
        if features[8] < 10:  # Too few nodes
            suggestions.append(
                "Code appears too minimal - consider adding documentation or structure")

    if not suggestions:
        suggestions.append("Code looks good! Keep up the good practices.")

    return suggestions


def predict(ast_json):
    global quality_classifier, naming_classifier, style_classifier, scaler, is_trained

    if not is_trained:
        raise ValueError("Model must be trained before making predictions")

    # Extract features
    features = extract_features_from_ast(ast_json)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Make predictions
    quality_pred = quality_classifier.predict(features_scaled)[0]
    naming_pred = naming_classifier.predict(features_scaled)[0]
    style_pred = style_classifier.predict(features_scaled)[0]

    # Generate suggestions based on predictions
    suggestions = _generate_suggestions(
        quality_pred, naming_pred, style_pred, features)

    return {
        'quality': quality_pred,
        'naming': naming_pred,
        'style': style_pred,
        'suggestions': suggestions
    }

if __name__ == "__main__":
    # Main execution code
    train_filepath = os.path.join(DATA_DIR, TRAIN_FILENAME)
    train_data = load_py150_jsonl(train_filepath, max_samples=MAX_SAMPLES)

    if train_data:
        train_models(train_data)
    else:
        print("Skipping training because no data was loaded. Check file paths and names.")

    if is_trained:
        save_model(MODEL_PATH)
    else:
        print("Model not trained. Skipping save.")

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        load_model(MODEL_PATH)
        eval_size = 50000
        eval_filepath = os.path.join(DATA_DIR, EVAL_FILENAME)
        if os.path.exists(eval_filepath):
            print(
                f"\n==Critiquing {eval_size} samples from the evaluation set:==")
            with open(eval_filepath, "r") as f:
                for i, line in enumerate(f):
                    if i >= eval_size:
                        break
                    try:
                        # Read each code sample
                        ast_json = json.loads(line)
                        # Store it in result
                        result = predict(ast_json)
                        # Print category and suggestion
                        print(f"\n--- Sample {i+1} ---")
                        print(f" Quality: {result['quality']}")
                        print(f" Naming: {result['naming']}")
                        print(f" Style: {result['style']}")
                        print(" Suggestions:")
                        for suggestion in result['suggestions']:
                            print(f" - {suggestion}")
                    except json.JSONDecodeError:
                        continue
        else:
            print(
                f"Evaluation file not found at {eval_filepath}. Cannot run inference.")
    else:
        print("Saved model not found. Please run the training and saving cells first.")
