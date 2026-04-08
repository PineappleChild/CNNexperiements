from PIL import Image
import utility_methods as utilMethods
import numpy as np
import matplotlib.pyplot as plt

import convolution as CONV
import fully_connected_nn as FCNN
import testing_matricies as testMat

# A LeNet inspired CNN for image classification on cifar-10
#conv1 -> relu1 -> mp1 -> conv2 -> relu2 -> mp2 -> flatten -> fc1 -> fc2 -> fc3 -> softmax for 10 classes

im = Image.open('../testing_images/hqdefault rere.jpg') 
pix = im.load()
print("(width x height x rgb(tuples) x kernels)")
smallerSize = im.size[1] if im.size[0] > im.size[1] else im.size[0]

print(smallerSize)
pixel_tuples_in_arr = []
for i in range(smallerSize):
    pixel_tuples_in_arr_row = []
    for j in range(smallerSize):
        pixel_tuples_in_arr_row.append(pix[i,j])
    pixel_tuples_in_arr.append(pixel_tuples_in_arr_row)

input_image_rgb = utilMethods.convert_rgb_image_to_cxhxw(pixel_tuples_in_arr)

input_value = input_image_rgb

conv_layer_one = CONV.convolution(num_kernels=16, kernel_size=5, stride=1, padding=utilMethods.compute_padding(len(input_value[0]), 5,1), filters=[utilMethods.gen_kernel(channels=3, size=5)] * 16, input_shape=None, bias=None)
filters_one = conv_layer_one.initialize_filters(input_depth=3, num_kernels=1, kernelHeight=5, kernelWidth=5)
convOutputOne = conv_layer_one.forward(input_value)

activatedChainOne = conv_layer_one.apply_ReLU(convOutputOne)

pooledMatOne = pooling.maxPooling([testMat.test_kernel_2x2] * 1, stride=1)
kernels_one = pooledMatOne.initialize_filters(input_depth=1, num_kernels=1, kernelHeight=2, kernelWidth=2)
pooledOutputOne = pooledMatOne.forward(activatedChainOne)

conv_layer_two = CONV.convolution(num_kernels=1, kernel_size=5, stride=1, padding=utilMethods.compute_padding(len(pooledOutputOne[0]), 5,1), filters=[utilMethods.gen_kernel(channels=1, size=5)] * 1, input_shape=None, bias=None)
filters_two = conv_layer_two.initialize_filters(input_depth=1, num_kernels=1, kernelHeight=5, kernelWidth=5)
convOutputTwo = conv_layer_two.forward(pooledOutputOne)

activatedChainTwo = conv_layer_two.apply_ReLU(convOutputTwo)

pooledMatTwo = pooling.maxPooling([testMat.test_kernel_2x2] * 1, stride=1)
kernels_two = pooledMatTwo.initialize_filters(input_depth=1, num_kernels=1, kernelHeight=2, kernelWidth=2)
pooledOutputTwo = pooledMatTwo.forward(activatedChainTwo)

fully_connected_one = FCNN.fully_connected(len(list(utilMethods.flatten(pooledOutputTwo))), 120)
final_fcc_one = fully_connected_one.forward(list(utilMethods.flatten(pooledOutputTwo)))

fully_connected_two = FCNN.fully_connected(len(final_fcc_one), 84)
final_fcc_two = fully_connected_two.forward(final_fcc_one)

fully_connected_three = FCNN.fully_connected(len(final_fcc_two), 10)
final_fcc_three = fully_connected_three.forward(final_fcc_two)

softmaxedList = utilMethods.softmax(final_fcc_three)
print(final_fcc_three)
print(softmaxedList)

CIFAR10_class_labels = ["airplane","automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print(CIFAR10_class_labels[softmaxedList.index(max(softmaxedList))])

def normalize_img(x):
    x = np.array(x)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-6:  
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn)
    return (x * 255).astype(np.uint8)


pixelsOG = input_value[0]
arrOG = normalize_img(pixelsOG)
arrOG = np.rot90(arrOG, k=3)

cnn1 = convOutputOne[0]
cnn1_arr = normalize_img(cnn1)
cnn1_arr = np.rot90(cnn1_arr, k=3)

relu1 = activatedChainOne[0]
relu1_arr = normalize_img(relu1)
relu1_arr = np.rot90(relu1_arr, k=3)

mpool1 = pooledOutputOne[0]
mpool1_arr = normalize_img(mpool1)
mpool1_arr = np.rot90(mpool1_arr, k=3)

cnn2 = convOutputTwo[0]
cnn2_arr = normalize_img(cnn2)
cnn2_arr = np.rot90(cnn2_arr, k=3)

relu2 = activatedChainTwo[0]
relu2_arr = normalize_img(relu2)
relu2_arr = np.rot90(relu2_arr, k=3)

mpool2 = pooledOutputTwo[0]
mpool2_arr = normalize_img(mpool2)
mpool2_arr = np.rot90(mpool2_arr, k=3)

f, axarr = plt.subplots(3, 3, figsize=(5, 8))

axarr[0][0].imshow(arrOG)
axarr[0][0].set_title("Original Image")
axarr[0][0].axis('off')

axarr[0][1].imshow(cnn1_arr)
axarr[0][1].set_title("[CNN1]")
axarr[0][1].axis('off')

axarr[0][2].imshow(relu1_arr)
axarr[0][2].set_title("[ReLU1]")
axarr[0][2].axis('off')

axarr[1][0].imshow(mpool1_arr)
axarr[1][0].set_title("[MaxPool1]")
axarr[1][0].axis('off')

axarr[1][1].imshow(cnn2_arr)
axarr[1][1].set_title("[CNN2]")
axarr[1][1].axis('off')

axarr[1][2].imshow(relu2_arr)
axarr[1][2].set_title("[ReLU2]")
axarr[1][2].axis('off')

axarr[2][0].imshow(mpool2_arr)
axarr[2][0].set_title("[MaxPool2]")
axarr[2][0].axis('off')

plt.tight_layout()
plt.show()
