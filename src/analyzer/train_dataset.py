import json
import numpy as np
from collections import defaultdict
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class MLCodeCriticAST:
    def __init__(self):
        self.quality_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.naming_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.style_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features_from_ast(self, ast_json):
        node_counts = defaultdict(int)
        names = []
        for node in ast_json:
            if isinstance(node, dict) and 'type' in node:
                node_type = node['type']
                node_counts[node_type] += 1
                if 'value' in node and isinstance(node['value'], str):
                    names.append(node['value'])
        total_nodes = len(ast_json)
        func_count = node_counts['FunctionDef']
        class_count = node_counts['ClassDef']
        assign_count = node_counts['Assign']
        import_count = node_counts['Import'] + node_counts['ImportFrom']
        if_count = node_counts['If']
        for_count = node_counts['For']
        while_count = node_counts['While']
        complexity = if_count + for_count + while_count
        name_len = np.mean([len(n) for n in names]) if names else 0
        snake_case = sum(1 for n in names if '_' in n and n.islower())
        camel_case = sum(1 for n in names if n != n.lower() and '_' not in n)
        call_count = node_counts['Call']
        attr_count = node_counts['attr']
        str_count = node_counts['Str']
        return_count = node_counts['Return']
        arguments_count = node_counts['arguments']
        body_count = node_counts['body']

        features = np.array([
            func_count, class_count, assign_count, import_count, complexity,
            node_counts['NameStore'], node_counts['NameLoad'], len(
                names), total_nodes,
            call_count, attr_count, str_count, return_count, arguments_count, body_count,
            name_len, snake_case, camel_case, 0, 0  # Padding for consistent feature count
        ], dtype=float)
        return features, names

    def load_py150_jsonl(self, filename, max_samples=10000):
        data = []
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                ast_json = json.loads(line)
                features, names = self.extract_features_from_ast(ast_json)
                # Dummy labeling: you should provide real scores!
                # For demo, we use simple heuristics for labels:
                quality = 60 + 10 * \
                    (features[0] > 0) + 10 * \
                    (features[1] > 0) - 10 * (features[4] > 10)
                naming = 60 + 10 * (features[15] > 6) - 10 * (features[16] < 2)
                style = 60 + 10 * (features[3] > 0) - 10 * (features[12] > 5)
                data.append({
                    "features": features,
                    "names": names,
                    "quality_score": min(100, max(0, quality)),
                    "naming_score": min(100, max(0, naming)),
                    "style_score": min(100, max(0, style))
                })
        return data

    def train_models(self, train_data):
        X = np.array([d["features"] for d in train_data])
        y_quality = [self._score_to_category(
            d["quality_score"]) for d in train_data]
        y_naming = [self._score_to_category(
            d["naming_score"]) for d in train_data]
        y_style = [self._score_to_category(
            d["style_score"]) for d in train_data]
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.quality_classifier.fit(X_scaled, y_quality)
        self.naming_classifier.fit(X_scaled, y_naming)
        self.style_classifier.fit(X_scaled, y_style)
        self.is_trained = True

    def predict_from_ast(self, ast_json):
        if not self.is_trained:
            return {"error": "Not trained"}
        features, names = self.extract_features_from_ast(ast_json)
        X_scaled = self.scaler.transform([features])
        quality_pred = self.quality_classifier.predict(X_scaled)[0]
        naming_pred = self.naming_classifier.predict(X_scaled)[0]
        style_pred = self.style_classifier.predict(X_scaled)[0]
        return {
            'overall_quality': quality_pred,
            'naming_quality': naming_pred,
            'style_quality': style_pred,
            'suggestions': self._generate_basic_suggestions(quality_pred, naming_pred, style_pred)
        }

    def _score_to_category(self, score):
        if score >= 80:
            return 'excellent'
        if score >= 60:
            return 'good'
        if score >= 40:
            return 'fair'
        return 'poor'

    def _generate_basic_suggestions(self, quality, naming, style):
        suggestions = []
        if quality in ['fair', 'poor']:
            suggestions.append("Improve code structure and logic.")
        if naming in ['fair', 'poor']:
            suggestions.append(
                "Use more descriptive and consistent naming conventions.")
        if style in ['fair', 'poor']:
            suggestions.append("Follow consistent code formatting and style.")
        return suggestions

    def save_models(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'quality': self.quality_classifier,
                'naming': self.naming_classifier,
                'style': self.style_classifier,
                'scaler': self.scaler,
                'trained': self.is_trained
            }, f)

    def load_models(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.quality_classifier = data['quality']
        self.naming_classifier = data['naming']
        self.style_classifier = data['style']
        self.scaler = data['scaler']
        self.is_trained = data['trained']


# Example usage
if __name__ == "__main__":
    # Train
    critic = MLCodeCriticAST()
    train_data = critic.load_py150_jsonl(
        "python100k_train.json", max_samples=50000)
    critic.train_models(train_data)
    critic.save_models("py150_model.pkl")

    # Test
    critic.load_models("py150_model.pkl")
    with open("python50k_eval.json", "r") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            ast_json = json.loads(line)
            result = critic.predict_from_ast(ast_json)
            print(f"Sample {i+1}:")
            print("Overall Quality:", result['overall_quality'])
            print("Naming Quality:", result['naming_quality'])
            print("Style Quality:", result['style_quality'])
            print("Suggestions:", result['suggestions'])
            print()
