import json
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


class SimpleMLCodeAnalyzer:
    def __init__(self):
        self.quality_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.naming_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.name_vectorizer = TfidfVectorizer(max_features=300)
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features_from_ast(self, ast_json):
        """Extract features from py150 AST JSON"""
        if not ast_json:
            return np.zeros(15)

        node_counts = defaultdict(int)
        names = []

        for node in ast_json:
            if isinstance(node, dict) and 'type' in node:
                node_counts[node['type']] += 1
                if 'value' in node and isinstance(node['value'], str):
                    names.append(node['value'])

        # Calculate features
        total_nodes = len(ast_json)
        complexity = node_counts['If'] + \
            node_counts['For'] + node_counts['While']

        features = np.array([
            node_counts['FunctionDef'],
            node_counts['ClassDef'],
            node_counts['Assign'],
            node_counts['ImportFrom'] + node_counts.get('Import', 0),
            complexity,
            node_counts['NameStore'],
            node_counts['NameLoad'],
            len(names),
            total_nodes,
            node_counts.get('Call', 0),
            node_counts.get('attr', 0),
            node_counts.get('Str', 0),
            node_counts.get('Return', 0),
            node_counts.get('arguments', 0),
            node_counts.get('body', 0)
        ], dtype=float)

        return features, names

    def generate_labels(self, ast_json):
        """Generate quality labels from AST"""
        node_counts = defaultdict(int)
        for node in ast_json:
            if isinstance(node, dict) and 'type' in node:
                node_counts[node['type']] += 1

        # Simple scoring
        quality = 60
        if node_counts['FunctionDef'] > 0:
            quality += 20
        if node_counts['ClassDef'] > 0:
            quality += 10
        if node_counts['If'] + node_counts['For'] > 10:
            quality -= 15

        naming = 60
        if node_counts['NameStore'] > node_counts['NameLoad'] * 0.3:
            naming += 15

        return min(100, quality), min(100, naming)

    def load_and_train(self, train_file, max_samples=5000):
        """Load JSON data and train models"""
        X_features = []
        X_names = []
        y_quality = []
        y_naming = []

        with open(train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    ast_json = json.loads(line.strip())
                    features, names = self.extract_features_from_ast(ast_json)
                    quality, naming = self.generate_labels(ast_json)

                    X_features.append(features)
                    X_names.append(' '.join(names))
                    y_quality.append(self.score_to_category(quality))
                    y_naming.append(self.score_to_category(naming))

                except:
                    continue

        # Train models
        X_features = self.scaler.fit_transform(X_features)
        X_names_vec = self.name_vectorizer.fit_transform(X_names).toarray()
        X_combined = np.concatenate([X_features, X_names_vec], axis=1)

        self.quality_classifier.fit(X_combined, y_quality)
        self.naming_classifier.fit(X_combined, y_naming)
        self.is_trained = True

    def predict_from_ast(self, ast_json):
        """Predict quality from AST JSON"""
        if not self.is_trained:
            return {"error": "Not trained"}

        features, names = self.extract_features_from_ast(ast_json)
        features_scaled = self.scaler.transform([features])
        names_vec = self.name_vectorizer.transform([' '.join(names)]).toarray()
        X_combined = np.concatenate([features_scaled, names_vec], axis=1)

        quality_pred = self.quality_classifier.predict(X_combined)[0]
        naming_pred = self.naming_classifier.predict(X_combined)[0]

        return {
            'quality': quality_pred,
            'naming': naming_pred,
            'suggestions': self.get_suggestions(quality_pred, naming_pred)
        }

    def score_to_category(self, score):
        if score >= 80:
            return 'excellent'
        if score >= 60:
            return 'good'
        if score >= 40:
            return 'fair'
        return 'poor'

    def get_suggestions(self, quality, naming):
        suggestions = []
        if quality in ['fair', 'poor']:
            suggestions.append("Improve code structure")
        if naming in ['fair', 'poor']:
            suggestions.append("Use better variable names")
        return suggestions

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'quality': self.quality_classifier,
                'naming': self.naming_classifier,
                'vectorizer': self.name_vectorizer,
                'scaler': self.scaler,
                'trained': self.is_trained
            }, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.quality_classifier = data['quality']
        self.naming_classifier = data['naming']
        self.name_vectorizer = data['vectorizer']
        self.scaler = data['scaler']
        self.is_trained = data['trained']

# Simple usage


def train_and_test():
    analyzer = SimpleMLCodeAnalyzer()

    # Train on your JSON file
    analyzer.load_and_train('python100k_train.json', max_samples=50000)
    analyzer.save_model('simple_model.pkl')

    # Test on eval data
    with open('python50k_eval.json', 'r') as f:
        for i, line in enumerate(f):
            if i >= 50:  # Test first 5 samples
                break
            try:
                ast_json = json.loads(line.strip())
                result = analyzer.predict_from_ast(ast_json)
                print(f"Sample {i+1}: {result}")
            except:
                continue


if __name__ == "__main__":
    train_and_test()
