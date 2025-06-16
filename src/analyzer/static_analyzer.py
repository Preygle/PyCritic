# Used to read Python code structure as a tree (AST = Abstract Syntax Tree)
import ast
import io
import sys
# Helps in capturing printed output or errors
from contextlib import redirect_stdout, redirect_stderr
# This checks if the code follows Python's style guide (PEP 8)
import pycodestyle
import subprocess
import tempfile  # Lets us create temporary files to work with
import os
from typing import Dict, List, Any  # Used to give hints about data types

# This class will help us check Python code for style, naming, structure, and complexity


class StaticAnalyzer:
    def __init__(self):
        # Setting up the PEP 8 style checker
        # Ignoring long lines for now (E501)
        self.style_guide = pycodestyle.StyleGuide(
            ignore=['E501'],
            max_line_length=88
        )

    # Main function to run all checks
    def analyze_code(self, code_string: str) -> Dict[str, Any]:
        return {
            'pep8_violations': self._check_pep8(code_string),
            'naming_issues': self._check_naming_conventions(code_string),
            'structure_issues': self._check_code_structure(code_string),
            'complexity_metrics': self._calculate_complexity(code_string)
        }

    # Check if the code follows PEP 8 (Python's official style guide)
    def _check_pep8(self, code_string: str) -> List[Dict]:
        violations = []

        # Saving code temporarily to run checks on it
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code_string)
            temp_file_path = temp_file.name

        try:
            # Running the checker
            checker = pycodestyle.Checker('', lines=code_string.splitlines())
            checker.check_all()

            # Storing the results
            for error in checker.results:
                violations.append({
                    'line': error[0],
                    'column': error[1],
                    'code': error[2],
                    'message': error[3],
                    'severity': 'error' if error[2].startswith('E') else 'warning'
                })

        finally:
            # Cleaning up the temporary file
            os.unlink(temp_file_path)

        return violations

    # This checks if functions, variables, and classes are named properly
    def _check_naming_conventions(self, code_string: str) -> List[Dict]:
        naming_issues = []

        try:
            # AST is a way to read Python code like a tree where each part (function, class, etc.) is a node
            tree = ast.parse(code_string)

            # Walk through the code using a custom visitor
            visitor = NamingConventionVisitor()
            visitor.visit(tree)
            naming_issues = visitor.issues
        except SyntaxError as e:
            naming_issues.append({
                'type': 'syntax_error',
                'line': e.lineno,
                'message': str(e),
                'severity': 'error'
            })

        return naming_issues

    # This checks how the code is organized
    def _check_code_structure(self, code_string: str) -> List[Dict]:
        structure_issues = []

        try:
            tree = ast.parse(code_string)

            # Check if imports are written at the top
            imports_at_top = self._check_imports_location(tree)
            if not imports_at_top:
                structure_issues.append({
                    'type': 'import_organization',
                    'message': 'Imports should be at the top of the file',
                    'severity': 'warning'
                })

            # Check if functions or classes are placed nicely
            structure_issues.extend(self._check_function_class_order(tree))

        except SyntaxError:
            pass  # Already caught in naming issues

        return structure_issues

    # This will try to calculate some basic complexity info
    def _calculate_complexity(self, code_string: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code_string)

            return {
                # Not implemented yet
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree),
                # Also needs implementation
                'nesting_depth': self._calculate_max_nesting_depth(tree),
                'function_count': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'class_count': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'lines_of_code': len(code_string.splitlines())
            }

        except SyntaxError:
            return {'error': 'Unable to parse code for complexity analysis'}

# This class walks through code and checks naming styles using AST


class NamingConventionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.issues = []

    def visit_FunctionDef(self, node):
        # Function names should be in snake_case like `my_function`
        if not self._is_snake_case(node.name):
            self.issues.append({
                'type': 'function_naming',
                'line': node.lineno,
                'name': node.name,
                'message': f'Function "{node.name}" should use snake_case naming',
                'suggestion': self._to_snake_case(node.name),
                'severity': 'warning'
            })

        # Check names of parameters inside the function
        for arg in node.args.args:
            if not self._is_snake_case(arg.arg):
                self.issues.append({
                    'type': 'parameter_naming',
                    'line': node.lineno,
                    'name': arg.arg,
                    'message': f'Parameter "{arg.arg}" should use snake_case naming',
                    'suggestion': self._to_snake_case(arg.arg),
                    'severity': 'warning'
                })

        self.generic_visit(node)  # Continue checking inside the function

    def visit_ClassDef(self, node):
        # Class names should be in PascalCase like `MyClass`
        if not self._is_pascal_case(node.name):
            self.issues.append({
                'type': 'class_naming',
                'line': node.lineno,
                'name': node.name,
                'message': f'Class "{node.name}" should use PascalCase naming',
                'suggestion': self._to_pascal_case(node.name),
                'severity': 'warning'
            })

        self.generic_visit(node)

    def visit_Assign(self, node):
        # Check naming of variables
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                # If variable is a constant (all caps), it should have underscores
                if var_name.isupper():
                    if '_' not in var_name and len(var_name) > 1:
                        self.issues.append({
                            'type': 'constant_naming',
                            'line': node.lineno,
                            'name': var_name,
                            'message': f'Constant "{var_name}" should use UPPER_CASE with underscores',
                            'severity': 'info'
                        })
                else:
                    # Other variables should be in snake_case
                    if not self._is_snake_case(var_name):
                        self.issues.append({
                            'type': 'variable_naming',
                            'line': node.lineno,
                            'name': var_name,
                            'message': f'Variable "{var_name}" should use snake_case naming',
                            'suggestion': self._to_snake_case(var_name),
                            'severity': 'warning'
                        })

        self.generic_visit(node)

    # Helper function to check if a name is snake_case
    def _is_snake_case(self, name: str) -> bool:
        if name.startswith('_'):
            name = name[1:]  # remove leading underscore like `_temp`
        return name.islower() and ('_' in name or name.islower())

    # Helper to check if name is PascalCase
    def _is_pascal_case(self, name: str) -> bool:
        return name[0].isupper() and '_' not in name

    # Convert any name to snake_case (e.g., MyVar → my_var)
    def _to_snake_case(self, name: str) -> str:
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    # Convert any name to PascalCase (e.g., my_var → MyVar)
    def _to_pascal_case(self, name: str) -> str:
        return ''.join(word.capitalize() for word in name.split('_'))
