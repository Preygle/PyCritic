import ast
import re


class StaticCodeAnalyzer:
    def __init__(self):
        pass

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


# Example usage
if __name__ == "__main__":
    analyzer = StaticCodeAnalyzer()
    test_code = """
class ProperClass:
    def correct_method(self):
        proper_variable = 42
        MAX_vALUE = 100
        _private_var = "secret"
        
        return proper_variable * MAX_VALUE
"""
    result = analyzer.check_naming_conventions(test_code)
    for violation in result['violations']:
        print(f"Line {violation['line']}: {violation['explanation']}")
        print(f"  Original: {violation['original']}")
        if violation['corrected']:
            print(f"  Suggested: {violation['corrected']}")
        print()
