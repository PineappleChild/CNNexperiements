import copy
import math
from numpy import random


class maxPooling:
    def __init__(self, kernels, stride):
        self.kernels = kernels
        self.stride = stride

    def forward(self, input):
        mat = input
        max_pool_mat_volume = []

        stride = self.stride
        kernel_size = len(self.kernels[0][0]) 

        for channel_idx, channel in enumerate(mat):
            H = len(channel)
            W = len(channel[0])
            out_H = ((H - kernel_size) // stride) + 1
            out_W = ((W - kernel_size) // stride) + 1

            max_pool_channel = []

            for i in range(out_H):
                row_output = []
                for j in range(out_W):
                    row_pos = i * stride
                    col_pos = j * stride

                    window = [channel[row_pos + ki][col_pos:col_pos + kernel_size] for ki in range(kernel_size)]
                    
                    max_val = max([val for row in window for val in row])
                    row_output.append(max_val)

                max_pool_channel.append(row_output)

            max_pool_mat_volume.append(max_pool_channel)

        print(
            f"[MAX POOL] channels: {len(mat)}, input: {H}x{W}, output: {len(max_pool_mat_volume[0])}x{len(max_pool_mat_volume[0][0])}")
        return max_pool_mat_volume

    def backwards(self):
        return None

    def initialize_filters(self, input_depth, num_kernels, kernelHeight, kernelWidth):
        filters = []

        fan_in = input_depth * kernelHeight * kernelWidth
        #he
        std = math.sqrt(2 / fan_in)  

        for _ in range(num_kernels):
            # input_depth, kH, kW
            new_filter = random.normal(0, std, size=(input_depth, kernelHeight, kernelWidth))
            filters.append(new_filter.tolist())

        self.filters = filters

        return filters
