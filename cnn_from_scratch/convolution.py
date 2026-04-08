import utility_methods as utilMethods
import copy
import math
from numpy import random

class convolution:
    def __init__(self, num_kernels, kernel_size, stride, padding, filters, input_shape, bias):
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filters = filters
        self.input_shape = input_shape
        self.bias = bias

    def forward(self, inputValue):
        mat = inputValue
        stride = self.stride
        padding = self.padding
        num_kernels = self.num_kernels
        output_volume = []
        
        # num of input channels (this case rgb)
        #row
        #col
        input_channels = len(mat)  
        input_height = len(mat[0]) 
        input_width = len(mat[0][0]) 

        for kernel in self.filters:
            if len(kernel) != input_channels:
                raise ValueError("Each kernel must have same depth as input channels")
            kernel_height = len(kernel[0])
            kernel_width = len(kernel[0][0])
            if kernel_height % 2 == 0 or kernel_width % 2 == 0:
                raise ValueError("Use odd kernel dimensions for center pixel alignment")

        # padding part
        padded_input = []
        for c in range(input_channels):
            padded_channel = [[0] * (input_width + 2 * padding) for _ in range(padding)]
            for row in mat[c]:
                padded_channel.append([0] * padding + row + [0] * padding)
            padded_channel += [[0] * (input_width + 2 * padding) for _ in range(padding)]
            padded_input.append(padded_channel)

        height_padded_dim = len(padded_input[0])
        width_padded_dim = len(padded_input[0][0])

        out_height_dim = ((height_padded_dim - kernel_height) // stride) + 1
        out_width_dim = ((width_padded_dim - kernel_width) // stride) + 1

        for kernel_idx, kernel in enumerate(self.filters):
            kernel_output = []
            for row in range(out_height_dim):
                row_output = []
                for col in range(out_width_dim):
                    conv_sum = 0
                    for channels in range(input_channels):
                        for kernelRow in range(kernel_height):
                            for kernelCol in range(kernel_width):
                                conv_sum += padded_input[channels][row * stride + kernelRow][col * stride + kernelCol] * kernel[channels][kernelRow][kernelCol]
                    row_output.append(conv_sum)
                kernel_output.append(row_output)
            output_volume.append(kernel_output)

        print(f"[CONV] input size: {input_height}x{input_width}x{input_channels}, output size: {out_height_dim}x{out_width_dim}, kernel count: {num_kernels}")
        return output_volume

    def apply_ReLU(self, target):
        relu_mat_kernel = []

        for kernel in target:
            relu_mat = []
            for row in kernel:
                relu_row = [max(0, val) for val in row] 
                relu_mat.append(relu_row)
            relu_mat_kernel.append(relu_mat)

        print("[ReLU] kernel count:", len(target), "output size:", len(relu_mat), "x", len(relu_mat[0]))
        return relu_mat_kernel

    def apply_sigmoid(self, target):
        import math

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        sigmoid_mat_kernel = []

        for kernel in target:
            sigmoid_mat = []
            for row in kernel:
                sigmoid_row = [int(sigmoid(val) * 255) for val in row]  
                sigmoid_mat.append(sigmoid_row)
            sigmoid_mat_kernel.append(sigmoid_mat)

        print("[Sigmoid] kernel count:", len(target), "output size:", len(sigmoid_mat), "x", len(sigmoid_mat[0]))
        return sigmoid_mat_kernel

    def backwards(self):
        return None


    def initialize_filters(self, input_depth, num_kernels, kernelHeight, kernelWidth):
        filters = []

        fan_in = input_depth * kernelHeight * kernelWidth
        #He init
        std = math.sqrt(2 / fan_in)  

        for _ in range(num_kernels):
            # input_depth, kH, kW
            new_filter = random.normal(0, std, size=(input_depth, kernelHeight, kernelWidth))
            filters.append(new_filter.tolist())

        self.filters = filters

        return filters


