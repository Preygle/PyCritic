
# Sample code for testing the PyCritic analysis

CONSTANT_vALUE = 42

def Calculate_sum(a, b):
    # This function is not very good
    return a + b

import pandas as np # Incorrect alias

class my_class:
    def __init__(self):
        self.someValue = 0
    
    def ADD(self, x):
        self.someValue += x
        self.someValue += CONSTANT_vALUE
        return self.someValue

def another_bad_function(SomeArgument):
    print(f"Argument was: {SomeArgument}")

