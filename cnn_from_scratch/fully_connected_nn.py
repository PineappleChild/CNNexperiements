import math
from numpy import random
import utility_methods as utilMethods


class fully_connected:
    def __init__(self, input_size, output_size):
        self.weights = random.randn(input_size, output_size) * math.sqrt(2.0 / input_size)
        self.biases = [25]*output_size

    def apply_activation_relu(self, input):
        return max(0, input)

    def forward(self, input):

        self.x = input

        if type(self.x[0]) == list:
            print(f"[FCForwardPass] matrix1: ({len(self.x)}, {len(self.x[0])}) matrix2: ({len(self.weights)}, {len(self.weights[0])})")
        else:
            print(f"[FCForwardPass] matrix1: (1, {len(self.x)}) matrix2: ({len(self.weights)}, {len(self.weights[0])})")

        dot_prod = utilMethods.dot(self.x, self.weights)

        for row in range(len(dot_prod)):
            dot_prod[row] += self.biases[row]

        for row in range(len(dot_prod)):
            dot_prod[row] = self.apply_activation_relu(dot_prod[row])

        return dot_prod