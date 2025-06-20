import ast
from static_analyzer import StaticCodeAnalyzer
code = """
class ProperClass:
    def correct_method(self):
        proper_variable = 42
        MAX_vALUE = 100
        _private_var = "secret"
        
        return proper_variable * MAX_VALUE
"""


def code_to_ast(code):
    # convert the code string to an AST
    return ast.parse(code)

def static_analyzer(code):
    analyzer = StaticCodeAnalyzer()
    return analyzer.check_naming_conventions(code)

def static_results_to_string(results):
    output = ""
    for violation in results['violations']:
        output += f"Line {violation['line']}: {violation['explanation']}\n"
        output += f"  Original: {violation['original']}\n"
        output += f"  Corrected: {violation['corrected'] if violation['corrected'] else 'No correction needed'}\n"
    output += f"\nTotal violations found: {len(results['violations'])}"
    return output

def analyze_code(code):
    results = static_analyzer(code)
    static_result = static_results_to_string(results)
    print(static_result)


print(analyze_code(code))
# print("Original code:")
# print(code)
# print("\nAST representation:")
# ast_representation = code_to_ast(code)
# print(ast.dump(ast_representation, indent=4))





