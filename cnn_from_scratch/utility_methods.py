import math

def ReLU(value):
    if value < 0:
        return 0
    return value


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    else:
        z = math.exp(value)
        return z / (1 + z)

def softmax(values):
    max_val = max(values)
    shifted = [v - max_val for v in values]

    exp_vals = [math.exp(v) for v in shifted]
    sum_exp = sum(exp_vals)

    return [ev / sum_exp for ev in exp_vals]

def dot(a, b):
    if isinstance(a[0], (int, float)):
        a = [a]
    if isinstance(b[0], (int, float)):
        b = [[x] for x in b]

    rows1 = len(a)
    cols1 = len(a[0])
    rows2 = len(b)
    cols2 = len(b[0])

    if cols1 != rows2:
        raise ValueError(f"Incompatible shapes {rows1,cols1} and {rows2,cols2}")

    dot_prod_final = []
    for row in range(rows1):
        dot_prod_row = []
        for col in range(cols2):
            dot_prod_sum = 0
            for index in range(cols1):
                dot_prod_sum += a[row][index] * b[index][col]
            dot_prod_row.append(dot_prod_sum)
        dot_prod_final.append(dot_prod_row)

    if len(dot_prod_final) == 1:
        return dot_prod_final[0]
    return dot_prod_final

def flatten(l):
    for item in l:
        if isinstance(item, list):
            for i in flatten(item):
                yield i
        else:
            yield item

def flatten_for_rgb(l):
    list_of_rgb_tuples = list(flatten(l))
    flat = []
    for (r, g, b) in list_of_rgb_tuples:
        flat.extend([r, g, b])
    return flat


def convert_rgb_image_to_cxhxw(image):
    R = len(image)
    C = len(image[0])

    # Initialize empty channels
    red_channel = [[0 for _ in range(C)] for _ in range(R)]
    green_channel = [[0 for _ in range(C)] for _ in range(R)]
    blue_channel = [[0 for _ in range(C)] for _ in range(R)]

    for i in range(R):
        for j in range(C):
            r, g, b = image[i][j]
            red_channel[i][j] = r
            green_channel[i][j] = g
            blue_channel[i][j] = b

    return [red_channel, green_channel, blue_channel]


#compute_padding(len(input_value[0]), 3, 1)
def compute_padding(input_size, kernel_size, stride):
    padding = ((input_size - 1) * stride + kernel_size - input_size) / 2
    return int(padding)

def gen_kernel(channels, size):
    gen_kernel_channels = []
    for channel in range(channels):
        gen_kernel_row = []
        for row in range(size):
            gen_kernel_col=[]
            for col in range(size):
                gen_kernel_col.append(0)
            gen_kernel_row.append(gen_kernel_col)
        gen_kernel_channels.append(gen_kernel_row)
    return gen_kernel_channels