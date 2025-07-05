# Sample code for testing. Contains all possible issues that the analyzer should catch.

CONSTANT_VALUE = 42


def calculate_sum(number_1,number_2):
    return number_1 + number_2


class MyClass:
    def __init__(self):
        self.some_value = 0

    def add(self, value):
        self.some_value += value
        self.some_value += CONSTANT_VALUE
        return self.some_value


def another_bad_function(some_argument):
    print(f"Argument was: {some_argument}")
