import ast
import re
from collections import defaultdict
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class MLCodeAnalyzer:
    def __init__(self):
        # Initialize classifiers with reduced complexity
        self.naming_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.quality_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.style_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)

        # Simplified text vectorizers
        self.name_vectorizer = TfidfVectorizer(max_features=500)
        self.code_vectorizer = TfidfVectorizer(max_features=1000)

        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_ml_features(self, code_string: str) -> np.ndarray:
        """Extract numerical features for ML models using simpler approaches"""

        lines = code_string.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            return np.zeros(20)  # Reduced feature count

        # Basic code metrics
        line_count = len(lines)
        non_empty_count = len(non_empty_lines)
        avg_line_len = sum(len(line) for line in lines) / \
            line_count if line_count else 0
        max_line_len = max(len(line) for line in lines) if lines else 0
        blank_lines = sum(1 for line in lines if not line.strip())

        # AST-based features
        functions = [n for n in ast.walk(
            tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        variables = self._extract_variables(tree)

        func_count = len(functions)
        class_count = len(classes)
        var_count = len(variables)
        complexity = self._calculate_cyclomatic_complexity(tree)
        nesting = self._calculate_nesting_depth(tree)

        # Naming pattern features
        all_names = [var['name']
                     for var in variables] + [f.name for f in functions]
        name_len = sum(len(name) for name in all_names) / \
            len(all_names) if all_names else 0
        snake_case = sum(
            1 for name in all_names if '_' in name and name.islower())
        camel_case = sum(1 for name in all_names if name !=
                         name.lower() and '_' not in name)

        # Code style features
        operators = sum(code_string.count(op)
                        for op in ['+', '-', '*', '/', '%', '=', '<', '>'])
        keywords = sum(code_string.count(kw)
                       for kw in ['if', 'else', 'for', 'while', 'def', 'class'])
        comments = sum(1 for line in lines if line.strip().startswith('#'))
        comment_ratio = comments / line_count if line_count else 0

        # Combine all features into a numpy array
        features = np.array([
            line_count, non_empty_count, avg_line_len, max_line_len, blank_lines,
            func_count, class_count, var_count, complexity, nesting,
            name_len, snake_case, camel_case,
            operators, keywords, comment_ratio,
            0, 0, 0, 0  # Padding for consistent feature count
        ], dtype=float)

        return features

    def train_models(self, training_data):
        """Train ML models with simplified data processing"""

        print("Training ML models...")

        # Prepare training data with basic lists
        X_features = []
        X_names = []
        X_code_text = []
        y_quality = []
        y_naming = []
        y_style = []

        for sample in training_data:
            code = sample['code']

            # Extract features
            features = self.extract_ml_features(code)
            X_features.append(features)

            # Extract names as text
            names = self._extract_all_names(code)
            X_names.append(' '.join(names))
            X_code_text.append(code)

            # Get labels
            y_quality.append(sample.get('quality_score', 0))
            y_naming.append(sample.get('naming_score', 0))
            y_style.append(sample.get('style_score', 0))

        # Convert to numpy arrays
        X_features = np.array(X_features)
        X_features_scaled = self.scaler.fit_transform(X_features)

        # Vectorize text features
        X_names_vec = self.name_vectorizer.fit_transform(X_names).toarray()
        X_code_vec = self.code_vectorizer.fit_transform(X_code_text).toarray()

        # Combine features
        X_combined = np.concatenate(
            [X_features_scaled, X_names_vec, X_code_vec], axis=1)

        # Convert scores to categories
        y_quality_cat = [self._score_to_category(score) for score in y_quality]
        y_naming_cat = [self._score_to_category(score) for score in y_naming]
        y_style_cat = [self._score_to_category(score) for score in y_style]

        # Train models
        self.quality_classifier.fit(X_combined, y_quality_cat)
        self.naming_classifier.fit(X_combined, y_naming_cat)
        self.style_classifier.fit(X_combined, y_style_cat)

        self.is_trained = True
        print("Training complete!")

    def predict_code_quality(self, code_string: str) -> dict:
        """Make predictions with simplified output"""

        if not self.is_trained:
            return {"error": "Models not trained yet"}

        # Extract features
        features = self.extract_ml_features(code_string)
        features_scaled = self.scaler.transform([features])

        # Extract text features
        names = self._extract_all_names(code_string)
        names_vec = self.name_vectorizer.transform([' '.join(names)]).toarray()
        code_vec = self.code_vectorizer.transform([code_string]).toarray()

        # Combine features
        X_combined = np.concatenate(
            [features_scaled, names_vec, code_vec], axis=1)

        # Make predictions
        quality_pred = self.quality_classifier.predict(X_combined)[0]
        naming_pred = self.naming_classifier.predict(X_combined)[0]
        style_pred = self.style_classifier.predict(X_combined)[0]

        return {
            'overall_quality': quality_pred,
            'naming_quality': naming_pred,
            'style_quality': style_pred,
            'suggestions': self._generate_basic_suggestions(quality_pred, naming_pred, style_pred)
        }

    # Simplified helper methods
    def _extract_variables(self, tree: ast.AST) -> list:
        """Simplified variable extraction"""
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(
                            {'name': target.id, 'line': node.lineno})
        return variables

    def _extract_all_names(self, code: str) -> list:
        """Extract all names from code"""
        names = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.append(node.id)
                elif isinstance(node, ast.FunctionDef):
                    names.append(node.name)
        except SyntaxError:
            pass
        return names

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Simplified complexity calculation"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
        return complexity

    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Simplified nesting depth calculation"""
        max_depth = 0
        stack = [(tree, 0)]

        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)

            if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                new_depth = depth + 1
                for child in ast.iter_child_nodes(node):
                    stack.append((child, new_depth))
            else:
                for child in ast.iter_child_nodes(node):
                    stack.append((child, depth))

        return max_depth

    def _score_to_category(self, score: int) -> str:
        """Convert score to category"""
        if score >= 80:
            return 'excellent'
        if score >= 60:
            return 'good'
        if score >= 40:
            return 'fair'
        return 'poor'

    def _generate_basic_suggestions(self, quality, naming, style) -> list:
        """Generate simple suggestions based on predictions"""
        suggestions = []
        if quality in ['fair', 'poor']:
            suggestions.append("Improve code structure and logic")
        if naming in ['fair', 'poor']:
            suggestions.append("Use more descriptive naming conventions")
        if style in ['fair', 'poor']:
            suggestions.append("Follow consistent coding style")
        return suggestions

    def save_models(self, filepath: str):
        """Save models to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'quality': self.quality_classifier,
                'naming': self.naming_classifier,
                'style': self.style_classifier,
                'name_vec': self.name_vectorizer,
                'code_vec': self.code_vectorizer,
                'scaler': self.scaler,
                'trained': self.is_trained
            }, f)

    def load_models(self, filepath: str):
        """Load models from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.quality_classifier = data['quality']
        self.naming_classifier = data['naming']
        self.style_classifier = data['style']
        self.name_vectorizer = data['name_vec']
        self.code_vectorizer = data['code_vec']
        self.scaler = data['scaler']
        self.is_trained = data['trained']
