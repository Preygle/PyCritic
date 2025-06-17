import ast
import re
from collections import defaultdict
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class MLCodeAnalyzer:
    def __init__(self):
        self.naming_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.quality_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.style_classifier = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.name_vectorizer = TfidfVectorizer(max_features=500)
        self.code_vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_ml_features(self, code_string: str) -> np.ndarray:
        lines = code_string.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            return np.zeros(20)
        line_count = len(lines)
        non_empty_count = len(non_empty_lines)
        avg_line_len = sum(len(line) for line in lines) / \
            line_count if line_count else 0
        max_line_len = max(len(line) for line in lines) if lines else 0
        blank_lines = sum(1 for line in lines if not line.strip())
        functions = [n for n in ast.walk(
            tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        variables = self._extract_variables(tree)
        func_count = len(functions)
        class_count = len(classes)
        var_count = len(variables)
        complexity = self._calculate_cyclomatic_complexity(tree)
        nesting = self._calculate_nesting_depth(tree)
        all_names = [var['name']
                     for var in variables] + [f.name for f in functions]
        name_len = sum(len(name) for name in all_names) / \
            len(all_names) if all_names else 0
        snake_case = sum(
            1 for name in all_names if '_' in name and name.islower())
        camel_case = sum(1 for name in all_names if name !=
                         name.lower() and '_' not in name)
        operators = sum(code_string.count(op)
                        for op in ['+', '-', '*', '/', '%', '=', '<', '>'])
        keywords = sum(code_string.count(kw)
                       for kw in ['if', 'else', 'for', 'while', 'def', 'class'])
        comments = sum(1 for line in lines if line.strip().startswith('#'))
        comment_ratio = comments / line_count if line_count else 0
        features = np.array([
            line_count, non_empty_count, avg_line_len, max_line_len, blank_lines,
            func_count, class_count, var_count, complexity, nesting,
            name_len, snake_case, camel_case,
            operators, keywords, comment_ratio,
            0, 0, 0, 0
        ], dtype=float)
        return features

    def train_models(self, training_data):
        X_features = []
        X_names = []
        X_code_text = []
        y_quality = []
        y_naming = []
        y_style = []
        for sample in training_data:
            code = sample['code']
            features = self.extract_ml_features(code)
            X_features.append(features)
            names = self._extract_all_names(code)
            X_names.append(' '.join(names))
            X_code_text.append(code)
            y_quality.append(sample.get('quality_score', 0))
            y_naming.append(sample.get('naming_score', 0))
            y_style.append(sample.get('style_score', 0))
        X_features = np.array(X_features)
        X_features_scaled = self.scaler.fit_transform(X_features)
        X_names_vec = self.name_vectorizer.fit_transform(X_names).toarray()
        X_code_vec = self.code_vectorizer.fit_transform(X_code_text).toarray()
        X_combined = np.concatenate(
            [X_features_scaled, X_names_vec, X_code_vec], axis=1)
        y_quality_cat = [self._score_to_category(score) for score in y_quality]
        y_naming_cat = [self._score_to_category(score) for score in y_naming]
        y_style_cat = [self._score_to_category(score) for score in y_style]
        self.quality_classifier.fit(X_combined, y_quality_cat)
        self.naming_classifier.fit(X_combined, y_naming_cat)
        self.style_classifier.fit(X_combined, y_style_cat)
        self.is_trained = True

    def predict_code_quality(self, code_string: str) -> dict:
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        features = self.extract_ml_features(code_string)
        features_scaled = self.scaler.transform([features])
        names = self._extract_all_names(code_string)
        names_vec = self.name_vectorizer.transform([' '.join(names)]).toarray()
        code_vec = self.code_vectorizer.transform([code_string]).toarray()
        X_combined = np.concatenate(
            [features_scaled, names_vec, code_vec], axis=1)
        quality_pred = self.quality_classifier.predict(X_combined)[0]
        naming_pred = self.naming_classifier.predict(X_combined)[0]
        style_pred = self.style_classifier.predict(X_combined)[0]
        naming_analysis = self.check_naming_conventions(code_string)
        return {
            'overall_quality': quality_pred,
            'naming_quality': naming_pred,
            'style_quality': style_pred,
            'naming_analysis': naming_analysis,
            'suggestions': self._generate_basic_suggestions(quality_pred, naming_pred, style_pred, naming_analysis)
        }

    def _extract_variables(self, tree: ast.AST) -> list:
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(
                            {'name': target.id, 'line': node.lineno})
        return variables

    def _extract_all_names(self, code: str) -> list:
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
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
        return complexity

    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
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
        if score >= 80:
            return 'excellent'
        if score >= 60:
            return 'good'
        if score >= 40:
            return 'fair'
        return 'poor'

    def _generate_basic_suggestions(self, quality, naming, style, naming_analysis) -> list:
        suggestions = []
        if quality in ['fair', 'poor']:
            suggestions.append("Improve code structure and logic")
        if naming in ['fair', 'poor']:
            suggestions.append("Use more descriptive naming conventions")
        if style in ['fair', 'poor']:
            suggestions.append("Follow consistent coding style")
        if naming_analysis.get('violations'):
            for v in naming_analysis['violations']:
                msg = f"Line {v['line']}: {v['explanation']}\n  Original: {v['original']}"
                if v['corrected']:
                    msg += f"\n  Suggested: {v['corrected']}"
                suggestions.append(msg)
        return suggestions

    def check_naming_conventions(self, code_string: str) -> dict:
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            return {"error": "Cannot analyze due to syntax errors"}
        violations = []
        code_lines = code_string.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                violations.extend(
                    self._check_function_naming(node, code_lines))
            elif isinstance(node, ast.ClassDef):
                violations.extend(self._check_class_naming(node, code_lines))
            elif isinstance(node, ast.Assign):
                violations.extend(
                    self._check_variable_naming(node, code_lines))
        return {'violations': violations}

    def _check_function_naming(self, node: ast.FunctionDef, code_lines) -> list:
        violations = []
        name = node.name
        line = self._get_line_from_code(code_lines, node.lineno)
        if name.startswith('__') and name.endswith('__'):
            return violations
        if not re.match(r'^[a-z_][a-z0-9_]*$', name):
            suggestion = self._to_snake_case(name)
            corrected = line.replace(name, suggestion, 1)
            violations.append({
                'type': 'function_naming',
                'line': node.lineno,
                'original': line,
                'corrected': corrected,
                'explanation': f"Function name '{name}' should be snake_case: '{suggestion}'"
            })
        if len(name) == 1:
            violations.append({
                'type': 'function_naming',
                'line': node.lineno,
                'original': line,
                'corrected': None,
                'explanation': "Avoid single-letter function names. Use something descriptive."
            })
        return violations

    def _check_class_naming(self, node: ast.ClassDef, code_lines) -> list:
        violations = []
        name = node.name
        line = self._get_line_from_code(code_lines, node.lineno)
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
            suggestion = self._to_pascal_case(name)
            corrected = line.replace(name, suggestion, 1)
            violations.append({
                'type': 'class_naming',
                'line': node.lineno,
                'original': line,
                'corrected': corrected,
                'explanation': f"Class name '{name}' should be CapWords/PascalCase: '{suggestion}'"
            })
        return violations

    def _check_variable_naming(self, node: ast.Assign, code_lines) -> list:
        violations = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                line = self._get_line_from_code(code_lines, node.lineno)
                if name.isupper() and not re.match(r'^[A-Z_][A-Z0-9_]*$', name):
                    suggestion = self._to_constant_case(name)
                    corrected = line.replace(name, suggestion, 1)
                    violations.append({
                        'type': 'constant_naming',
                        'line': node.lineno,
                        'original': line,
                        'corrected': corrected,
                        'explanation': f"Constant '{name}' should be ALL_CAPS: '{suggestion}'"
                    })
                elif not name.isupper() and not re.match(r'^[a-z_][a-z0-9_]*$', name):
                    suggestion = self._to_snake_case(name)
                    corrected = line.replace(name, suggestion, 1)
                    violations.append({
                        'type': 'variable_naming',
                        'line': node.lineno,
                        'original': line,
                        'corrected': corrected,
                        'explanation': f"Variable '{name}' should be snake_case: '{suggestion}'"
                    })
                if len(name) == 1 and name not in ['i', 'j', 'k', 'x', 'y', 'z']:
                    violations.append({
                        'type': 'variable_naming',
                        'line': node.lineno,
                        'original': line,
                        'corrected': None,
                        'explanation': f"Variable '{name}' is a single letter. Use a descriptive name."
                    })
        return violations

    def _get_line_from_code(self, code_lines, lineno):
        if 1 <= lineno <= len(code_lines):
            return code_lines[lineno - 1].strip()
        return ""

    def _to_snake_case(self, name: str) -> str:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower()

    def _to_pascal_case(self, name: str) -> str:
        return ''.join(word.capitalize() for word in re.split(r'_|-', name))

    def _to_constant_case(self, name: str) -> str:
        return self._to_snake_case(name).upper()

    def save_models(self, filepath: str):
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
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.quality_classifier = data['quality']
        self.naming_classifier = data['naming']
        self.style_classifier = data['style']
        self.name_vectorizer = data['name_vec']
        self.code_vectorizer = data['code_vec']
        self.scaler = data['scaler']
        self.is_trained = data['trained']


# Example usage
if __name__ == "__main__":
    analyzer = MLCodeAnalyzer()
    test_code = """
class networkProfileTab:
    def getUserData(self):
        userName = "test"
        MAX_limit = 100
        _helperVar = 5
        return userName

def a():
    x = 5
    return x
"""
    result = analyzer.check_naming_conventions(test_code)
    for violation in result['violations']:
        print(f"Line {violation['line']}: {violation['explanation']}")
        print(f"  Original: {violation['original']}")
        if violation['corrected']:
            print(f"  Suggested: {violation['corrected']}")
        print()
