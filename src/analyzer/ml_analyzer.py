# This file analyzes Python code using beginner-friendly logic and ML features (optional)
# It checks things like variable names, formatting, readability, and gives tips

import ast  # For understanding the structure of Python code
import re
from typing import Dict, List, Any, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os


class MLAnalyzer:
    def __init__(self):
        # ML model for checking naming (optional, not required for beginners)
        self.naming_model = None
        self.quality_model = None  # ML model for overall code quality
        self.is_trained = False
        self.feature_names = []  # Names of features used in ML model

    def analyze_code(self, code_string: str) -> Dict[str, Any]:
        """Main function to analyze code in a beginner-friendly way"""

        # Step 1: Extract useful information from code
        features = self._extract_simple_features(code_string)

        # Step 2: Use the features to generate feedback
        ml_results = {
            'readability_score': self._calculate_readability_score(features),
            'naming_quality': self._assess_naming_quality(code_string),
            'code_style_tips': self._generate_style_tips(features),
            'beginner_suggestions': self._get_beginner_suggestions(features),
            'good_practices_found': self._identify_good_practices(code_string)
        }

        return ml_results

    def _extract_simple_features(self, code_string: str) -> Dict[str, Any]:
        """Break down code and collect simple, understandable info"""

        lines = code_string.strip().split('\n')  # Break into lines
        # Ignore empty lines
        non_empty_lines = [line for line in lines if line.strip()]

        # Try parsing the code
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            # If the code has errors, return minimal info
            return {
                'total_lines': len(lines),
                'non_empty_lines': len(non_empty_lines),
                'has_syntax_error': True,
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
            }

        # If code is valid, calculate all beginner-friendly features
        features = {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'blank_lines': len(lines) - len(non_empty_lines),
            'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,

            'function_count': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            'class_count': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            'variable_count': len(self._get_variable_names(tree)),

            'descriptive_names_ratio': self._calculate_descriptive_names_ratio(tree),
            'snake_case_compliance': self._check_snake_case_compliance(tree),
            'meaningful_function_names': self._check_meaningful_function_names(tree),

            # How deep are if/else/loops
            'max_nesting_level': self._calculate_simple_nesting(tree),
            'has_comments': self._has_comments(code_string),
            'uses_good_spacing': self._check_spacing_patterns(code_string),

            'has_syntax_error': False,
            'uses_main_guard': self._has_main_guard(tree),
            'has_docstrings': self._has_docstrings(tree)
        }

        return features

    def _calculate_readability_score(self, features: Dict) -> Dict[str, Any]:
        """Give a score and tips for how easy the code is to read"""

        score = 0
        max_score = 100
        feedback = []

        # Check line length
        if features['avg_line_length'] <= 80:
            score += 20
            feedback.append("✓ Good line length - easy to read")
        else:
            feedback.append(
                "⚠ Lines are too long - try to keep under 80 characters")

        # Check naming
        if features['descriptive_names_ratio'] > 0.7:
            score += 25
            feedback.append("✓ Good use of descriptive variable names")
        elif features['descriptive_names_ratio'] > 0.4:
            score += 15
            feedback.append("⚠ Some variable names could be more descriptive")
        else:
            feedback.append(
                "⚠ Use more descriptive variable names (avoid x, y, temp)")

        # Check structure
        if features['function_count'] > 0:
            score += 10
            feedback.append("✓ Good use of functions")

        if features['has_comments']:
            score += 10
            feedback.append("✓ Good use of comments")
        else:
            feedback.append("⚠ Add comments to explain your code")

        # Spacing and formatting
        if features['uses_good_spacing']:
            score += 20
            feedback.append("✓ Good spacing and formatting")
        else:
            feedback.append(
                "⚠ Improve spacing around operators and after commas")

        # Bonus points for best practices
        if features['has_docstrings']:
            score += 8
            feedback.append("✓ Excellent! You're using docstrings")

        if features['uses_main_guard']:
            score += 7
            feedback.append("✓ Great! You're using if __name__ == '__main__'")

        return {
            'score': min(score, max_score),
            'grade': self._score_to_grade(score),
            'feedback': feedback,
            'areas_to_improve': self._get_improvement_areas(features)
        }

    def _assess_naming_quality(self, code_string: str) -> Dict[str, Any]:
        """Check if variable and function names are meaningful or generic"""

        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            return {'error': 'Cannot analyze naming due to syntax errors'}

        variable_names = self._get_variable_names(tree)
        function_names = [node.name for node in ast.walk(
            tree) if isinstance(node, ast.FunctionDef)]

        poor_names = ['x', 'y', 'z', 'temp', 'data',
                      'var', 'a', 'b', 'c', 'i', 'j', 'k']

        naming_issues = []
        good_names = []

        for name in variable_names + function_names:
            if name.lower() in poor_names:
                naming_issues.append({
                    'name': name,
                    'issue': 'Too generic - use descriptive names',
                    'suggestions': self._suggest_better_names(name)
                })
            elif len(name) >= 3 and '_' in name:
                good_names.append(name)

        return {
            'total_names': len(variable_names + function_names),
            'good_names': good_names,
            'naming_issues': naming_issues,
            'naming_score': max(0, 100 - (len(naming_issues) * 20))
        }

    def _generate_style_tips(self, features: Dict) -> List[str]:
        """Give basic tips to improve formatting and structure"""

        tips = []

        if features['avg_line_length'] > 100:
            tips.append(
                "💡 Tip: Break long lines into multiple shorter lines for better readability")

        if features['function_count'] == 0 and features['total_lines'] > 20:
            tips.append(
                "💡 Tip: Consider breaking your code into functions - it makes it easier to read and test")

        if not features['has_comments'] and features['total_lines'] > 10:
            tips.append(
                "💡 Tip: Add comments to explain what your code does - your future self will thank you!")

        if features['descriptive_names_ratio'] < 0.5:
            tips.append(
                "💡 Tip: Use descriptive variable names like 'student_name' instead of 'x'")

        if features['max_nesting_level'] > 3:
            tips.append(
                "💡 Tip: Try to reduce nesting levels - consider using functions or early returns")

        return tips

    def _get_beginner_suggestions(self, features: Dict) -> List[Dict[str, str]]:
        """Specific small tips based on what the code is missing"""

        suggestions = []

        if features['has_syntax_error']:
            suggestions.append({
                'priority': 'high',
                'category': 'Syntax',
                'suggestion': 'Fix syntax errors first - Python needs correct syntax to run',
                'example': 'Check for missing colons (:) after if statements and function definitions'
            })

        if not features['uses_good_spacing']:
            suggestions.append({
                'priority': 'medium',
                'category': 'Formatting',
                'suggestion': 'Add spaces around operators and after commas',
                'example': 'Write "x = 5 + 3" instead of "x=5+3"'
            })

        if features['descriptive_names_ratio'] < 0.6:
            suggestions.append({
                'priority': 'medium',
                'category': 'Naming',
                'suggestion': 'Use descriptive variable names that explain what the data represents',
                'example': 'Use "student_age" instead of "x" or "age"'
            })

        if not features['has_comments'] and features['total_lines'] > 15:
            suggestions.append({
                'priority': 'low',
                'category': 'Documentation',
                'suggestion': 'Add comments to explain complex parts of your code',
                'example': '# Calculate the average of all test scores'
            })

        return suggestions

    # Helper Functions Below

    def _get_variable_names(self, tree: ast.AST) -> List[str]:
        """Get all variable names used in the code"""
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.append(target.id)
        return names

    def _calculate_descriptive_names_ratio(self, tree: ast.AST) -> float:
        """Check what % of variable names are meaningful"""
        all_names = self._get_variable_names(tree)
        if not all_names:
            return 1.0

        generic_names = ['x', 'y', 'z', 'temp', 'data', 'var', 'a', 'b', 'c']
        descriptive_count = sum(1 for name in all_names if name.lower(
        ) not in generic_names and len(name) >= 3)

        return descriptive_count / len(all_names)

    def _score_to_grade(self, score: int) -> str:
        """Turn score into a grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    def _suggest_better_names(self, poor_name: str) -> List[str]:
        """Suggest better names for common bad variables"""
        suggestions = {
            'x': ['number', 'value', 'coordinate', 'input_value'],
            'y': ['result', 'output', 'coordinate', 'calculated_value'],
            'temp': ['temporary_value', 'current_item', 'placeholder'],
            'data': ['student_data', 'file_content', 'user_input'],
            'i': ['index', 'counter', 'position'],
            'j': ['inner_index', 'column', 'second_counter']
        }
        return suggestions.get(poor_name.lower(), ['descriptive_name', 'meaningful_variable'])


