
# Sample code for testing. Contains all possible issues that the analyzer should catch.

CONSTANT_vALUE = 42 # Incorrect constant naming, should be UPPER_SNAKE_CASE

def Calculate_sum(a, b): #func. name not in snake_case
    # This function is not very good
    return a + b #not descriptive variable names

import pandas as np # Incorrect alias SHOULD BE pd. Also, this import is not used anywhere in the code

class my_class: # class name not in PascalCase
    def __init__(self):
        self.someValue = 0
    
    def ADD(self, x): # Function name not in snake_case, SHOULD BE add
        self.someValue += x # not descriptive variable name
        self.someValue += CONSTANT_vALUE
        return self.someValue

def another_bad_function(SomeArgument):
    print(f"Argument was: {SomeArgument}")

