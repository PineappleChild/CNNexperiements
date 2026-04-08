import numpy as np
import os
import time
from PIL import Image
from cnn_from_scratch import utility_methods as utilMethods
from dataclasses import dataclass
import cupy as cp

#cupy is used for gpu utilization
#tried to implement something like vgg net
#image classification using cifar-10 dataset

# (conv -> bn -> relu) x2 -> mp -> dropout -> 
# (conv -> bn -> relu) x2 -> mp -> dropout -> 
# (conv -> bn -> relu) x3 -> mp -> dropout -> 
# (conv -> bn-> relu ) x2 -> mp -> dropout -> 
# gap -> fc -> relu -> dropout -> fc -> logits


class SoftmaxCrossEntropy:
    def forward(self, logits, target):
        self.batch_size = logits.shape[0]
        self.logits = logits

        logits_shifted = logits - cp.max(logits, axis=1, keepdims=True)
        
        exps = cp.exp(logits_shifted).astype(cp.float32)
        self.probs = exps / cp.sum(exps, axis=1, keepdims=True)

        correct_logprobs = -cp.log(
            self.probs[cp.arange(self.batch_size), target] + 1e-10
        )
        loss = cp.mean(correct_logprobs)
        self.target = target
        return loss

    def backward(self, smoothing=0.1):
        grad = self.probs.copy()
        grad[cp.arange(self.batch_size), self.target] -= 1.0
        grad += smoothing / (grad.shape[1] - 1)
        grad[cp.arange(self.batch_size), self.target] -= smoothing
        grad /= self.batch_size
        return grad.astype(cp.float32)


class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask


class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)


class GlobalAveragePooling:
    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        return cp.mean(x, axis=(2, 3)).astype(cp.float32)

    def backward(self, dout):
        N, C, H, W = self.x_shape
        dx = dout[:, :, None, None] / (H * W)
        return cp.broadcast_to(dx, (N, C, H, W)).astype(cp.float32)


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, x):
        if self.training and self.p > 0:
            self.mask = (
                    (cp.random.rand(*x.shape) > self.p)
                    .astype(cp.float32)
                    / (1.0 - self.p)
            )
            return x * self.mask
        return x

    def backward(self, d_out):
        if self.training and self.p > 0:
            return d_out * self.mask
        return d_out


class Convolution:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, backend=xp):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.backend = backend

        fan_in = in_channels * kernel_size * kernel_size
        self.filters = (
            backend.random.randn(out_channels, in_channels, kernel_size, kernel_size)
            .astype(backend.float32)
            * backend.sqrt(2.0 / fan_in)
        )
        self.bias = backend.zeros(out_channels, dtype=backend.float32)

    def im2col(self, x):
        B, C, H, W = x.shape
        kH, kW, stride = self.kernel_size, self.kernel_size, self.stride
        out_h = (H - kH) // stride + 1
        out_w = (W - kW) // stride + 1

        shape = (B, C, kH, kW, out_h, out_w)
        strides = (
            x.strides[0],
            x.strides[1],
            x.strides[2],
            x.strides[3],
            stride * x.strides[2],
            stride * x.strides[3],
        )
        patches = cp.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return patches.reshape(B, C * kH * kW, out_h * out_w)

    def col2im(self, cols, x_shape):
        B, C, H, W = x_shape
        kH, kW, stride = self.kernel_size, self.kernel_size, self.stride
        out_h = (H - kH) // stride + 1
        out_w = (W - kW) // stride + 1

        cols_reshaped = cols.reshape(B, C, kH, kW, out_h, out_w)
        dx = cp.zeros(x_shape, dtype=cols.dtype)

        
        for i in range(kH):
            for j in range(kW):
                dx[:, :, i:i + stride*out_h:stride, j:j + stride*out_w:stride] += cols_reshaped[:, :, i, j, :, :]
        return dx

    def forward(self, x):
        self.x = x
        if self.padding > 0:
            self.x_padded = cp.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
            )
        else:
            self.x_padded = x

        
        self.cols = self.im2col(self.x_padded)
        W_col = self.filters.reshape(self.filters.shape[0], -1)
        
        out = cp.matmul(W_col[None, :, :], self.cols) + self.bias[None, :, None]

        B, _, H_p, W_p = self.x_padded.shape
        out_h = (H_p - self.kernel_size) // self.stride + 1
        out_w = (W_p - self.kernel_size) // self.stride + 1

        return out.reshape(B, self.filters.shape[0], out_h, out_w).astype(cp.float32)

    def backward(self, d_out):
        B, OC, OH, OW = d_out.shape
        
        dout_flat = d_out.reshape(B, OC, -1)
        self.db = cp.sum(d_out, axis=(0, 2, 3)).astype(cp.float32)

        
        dW = cp.matmul(dout_flat, self.cols.transpose(0, 2, 1))
        self.dW = cp.sum(dW, axis=0).reshape(self.filters.shape).astype(cp.float32)

        W_col = self.filters.reshape(OC, -1)
        dx_cols = cp.matmul(W_col.T[None, :, :], dout_flat)
        dx_padded = self.col2im(dx_cols, self.x_padded.shape)

        if self.padding > 0:
            return dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dx_padded


class MaxPooling:
    def __init__(self, kernel_size=2, stride=2):
        self.k = kernel_size
        self.s = stride

    def forward(self, x):
        self.x = x
        B, C, H, W = x.shape
        out_h = (H - self.k) // self.s + 1
        out_w = (W - self.k) // self.s + 1

        shape = (B, C, out_h, out_w, self.k, self.k)
        strides = (
            x.strides[0],
            x.strides[1],
            self.s * x.strides[2],
            self.s * x.strides[3],
            x.strides[2],
            x.strides[3]
        )
        windows = cp.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        self.argmax = cp.argmax(windows.reshape(B, C, out_h, out_w, -1), axis=-1)

        out = cp.max(windows, axis=(4, 5))
        return out

    def backward(self, d_out):
        B, C, H, W = self.x.shape
        out_h, out_w = d_out.shape[2], d_out.shape[3]

        dx_windows = cp.zeros((B, C, out_h, out_w, self.k * self.k), dtype=cp.float32)

        dx_windows.reshape(B, C, out_h, out_w, self.k * self.k)[
            cp.arange(B)[:, None, None, None],
            cp.arange(C)[None, :, None, None],
            cp.arange(out_h)[None, None, :, None],
            cp.arange(out_w)[None, None, None, :],
            self.argmax
        ] = d_out

        dx = cp.zeros_like(self.x, dtype=cp.float32)
        for i in range(self.k):
            for j in range(self.k):
                dx[:, :, i:i + self.s * out_h:self.s, j:j + self.s * out_w:self.s] += dx_windows[:, :, :, :,
                                                                                      i * self.k + j]

        return dx


class FullyConnected:
    def __init__(self, in_features, out_features):
        self.weights = (
                cp.random.randn(in_features, out_features)
                .astype(cp.float32)
                * cp.sqrt(2.0 / in_features)
        )
        self.biases = cp.zeros(out_features, dtype=cp.float32)

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.biases

    def backward(self, d_out):
        self.dW = (self.x.T @ d_out).astype(cp.float32)
        self.db = cp.sum(d_out, axis=0).astype(cp.float32)
        return d_out @ self.weights.T


class BatchNorm:
    def __init__(self, num_features, momentum=0.9, eps=1e-5, backend=xp):
        self.backend = backend
        self.gamma = backend.ones(num_features, dtype=cp.float32)
        self.beta = backend.zeros(num_features, dtype=cp.float32)
        self.momentum = momentum
        self.eps = eps
        self.running_mean = backend.zeros(num_features, dtype=cp.float32)
        self.running_var = backend.ones(num_features, dtype=cp.float32)
        self.training = True

        self.mean_buffer = None
        self.var_buffer = None
        self.x_hat_buffer = None

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            self.mean_buffer = mean
            self.var_buffer = var

            self.x_hat_buffer = (x - mean) / self.backend.sqrt(var + self.eps)
        else:
            self.x_hat_buffer = (x - self.running_mean.reshape(1, -1, 1, 1)) / \
                                self.backend.sqrt(self.running_var.reshape(1, -1, 1, 1) + self.eps)

        if self.training:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()

        return (self.gamma.reshape(1, -1, 1, 1) * self.x_hat_buffer +
                self.beta.reshape(1, -1, 1, 1)).astype(cp.float32)

    def backward(self, dout):
        B, C, H, W = dout.shape
        N = B * H * W

        x_hat = self.x_hat_buffer
        var = self.var_buffer

        gamma_reshaped = self.gamma.reshape(1, C, 1, 1)
        dx_hat = dout * gamma_reshaped

        sum_dx_hat = dx_hat.sum(axis=(0, 2, 3), keepdims=True)
        sum_dx_hat_xhat = (dx_hat * x_hat).sum(axis=(0, 2, 3), keepdims=True)

        inv_std = 1. / self.backend.sqrt(var + self.eps)
        dx = (dx_hat - sum_dx_hat / N - x_hat * sum_dx_hat_xhat / N) * inv_std

        return dx.astype(cp.float32)


def clip_gradients(layer, max_norm=5.0):
    if hasattr(layer, 'dW'):
        norm = cp.linalg.norm(layer.dW)
        if norm > max_norm:
            layer.dW = layer.dW * (max_norm / norm)
    if hasattr(layer, 'db'):
        norm = cp.linalg.norm(layer.db)
        if norm > max_norm:
            layer.db = layer.db * (max_norm / norm)


# val and accuracy

def evaluate(layers, test_list, directory_test, num_samples):

    CIFAR10_class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    for lw in layers:
        if isinstance(lw.layer, Dropout):
            lw.layer.training = False

    correct = 0
    batch_size = num_samples // 10
    num_batches = num_samples // batch_size

    for _ in range(num_batches):
        idx = np.random.choice(len(test_list), size=batch_size, replace=False)
        batch_items = [test_list[i] for i in idx]

        images = []
        labels = []

        for e, f in batch_items:
            image_path = f"{directory_test}/{e.name}/{f.name}"
            pixel_values = convert_to_pixel_values(image_path).astype(np.float32)
            pixel_values = (pixel_values / 255.0 - 0.5) / 0.5
            images.append(pixel_values)
            labels.append(CIFAR10_class_labels.index(e.name))

        images = cp.array(images, dtype=cp.float32)
        labels = cp.array(labels, dtype=cp.int64)

        out = images
        for lw in layers:
            out = lw.layer.forward(out)

        preds = np.argmax(out, axis=1)
        correct += np.sum(preds == labels)

    for lw in layers:
        if isinstance(lw.layer, Dropout):
            lw.layer.training = True

    return correct / num_samples


loss_values_per_iter = []
val_accuracy_per_epoch = []

# schedulers

def get_lr(epoch, lr, total_epochs):
    return cosine_annealing(epoch, total_epochs, initial_lr=lr, min_lr=0.001)
    # return step_decay(epoch, initial_lr=lr, step_size=30, decay_factor=0.1)

def step_decay(epoch, initial_lr=0.01, step_size=30, decay_factor=0.1):
    lr = initial_lr * (decay_factor ** (epoch // step_size))
    return lr

def cosine_annealing(epoch, total_epochs, initial_lr=0.01, min_lr=0.001):
    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cp.cos(cp.pi * epoch / total_epochs))
    return lr

# training loop

@dataclass
class LayerWrapper:
    layer: object
    name: str

def run_nn_train(num_epochs, batch_size, initial_lr, weight_decay_amount, get_batch_fn, test_list, directory_test, verbose=True):
    conv1 = Convolution(3, 64, kernel_size=3, stride=1, padding=1)
    bn1 = BatchNorm(64)
    relu1 = ReLU()

    conv2 = Convolution(64, 64, kernel_size=3, stride=1, padding=1)
    bn2 = BatchNorm(64)
    relu2 = ReLU()
    pool1 = MaxPooling(kernel_size=2, stride=2)
    dropout1 = Dropout(p=0.1)

    conv3 = Convolution(64, 128, kernel_size=3, stride=1, padding=1)
    bn3 = BatchNorm(128)
    relu3 = ReLU()

    conv4 = Convolution(128, 128, kernel_size=3, stride=1, padding=1)
    bn4 = BatchNorm(128)
    relu4 = ReLU()
    pool2 = MaxPooling(kernel_size=2, stride=2)
    dropout2 = Dropout(p=0.2)

    conv5 = Convolution(128, 256, kernel_size=3, stride=1, padding=1)
    bn5 = BatchNorm(256)
    relu5 = ReLU()

    conv6 = Convolution(256, 256, kernel_size=3, stride=1, padding=1)
    bn6 = BatchNorm(256)
    relu6 = ReLU()

    conv7 = Convolution(256, 256, kernel_size=3, stride=1, padding=1)
    bn7 = BatchNorm(256)
    relu7 = ReLU()
    pool3 = MaxPooling(kernel_size=2, stride=2)
    dropout3 = Dropout(p=0.3)

    conv8 = Convolution(256, 512, kernel_size=3, stride=1, padding=1)
    bn8 = BatchNorm(512)
    relu8 = ReLU()

    conv9 = Convolution(512, 512, kernel_size=3, stride=1, padding=1)
    bn9 = BatchNorm(512)
    relu9 = ReLU()
    pool4 = MaxPooling(kernel_size=2, stride=2)
    dropout4 = Dropout(p=0.4)

    gap = GlobalAveragePooling()
    fc1 = FullyConnected(512, 512)
    relu_fc = ReLU()
    dropout5 = Dropout(p=0.5)
    fc2 = FullyConnected(512, 10)

    layers = [
        LayerWrapper(conv1, 'conv1'),
        LayerWrapper(bn1, 'bn1'),
        LayerWrapper(relu1, 'relu1'),

        LayerWrapper(conv2, 'conv2'),
        LayerWrapper(bn2, 'bn2'),
        LayerWrapper(relu2, 'relu2'),
        LayerWrapper(pool1, 'pool1'),
        LayerWrapper(dropout1, 'dropout1'),

        LayerWrapper(conv3, 'conv3'),
        LayerWrapper(bn3, 'bn3'),
        LayerWrapper(relu3, 'relu3'),

        LayerWrapper(conv4, 'conv4'),
        LayerWrapper(bn4, 'bn4'),
        LayerWrapper(relu4, 'relu4'),
        LayerWrapper(pool2, 'pool2'),
        LayerWrapper(dropout2, 'dropout2'),

        LayerWrapper(conv5, 'conv5'),
        LayerWrapper(bn5, 'bn5'),
        LayerWrapper(relu5, 'relu5'),

        LayerWrapper(conv6, 'conv6'),
        LayerWrapper(bn6, 'bn6'),
        LayerWrapper(relu6, 'relu6'),

        LayerWrapper(conv7, 'conv7'),
        LayerWrapper(bn7, 'bn7'),
        LayerWrapper(relu7, 'relu7'),
        LayerWrapper(pool3, 'pool3'),
        LayerWrapper(dropout3, 'dropout3'),

        LayerWrapper(conv8, 'conv8'),
        LayerWrapper(bn8, 'bn8'),
        LayerWrapper(relu8, 'relu8'),

        LayerWrapper(conv9, 'conv9'),
        LayerWrapper(bn9, 'bn9'),
        LayerWrapper(relu9, 'relu9'),
        LayerWrapper(pool4, 'pool4'),
        LayerWrapper(dropout4, 'dropout4'),

        LayerWrapper(gap, 'gap'),
        LayerWrapper(fc1, 'fc1'),
        LayerWrapper(relu_fc, 'relu_fc'),
        LayerWrapper(dropout5, 'dropout5'),
        LayerWrapper(fc2, 'fc2')
    ]

    criterion = SoftmaxCrossEntropy()
    best_val_acc = 0.0
    # floor div val for to train on % of train data
    steps_per_epoch = (len(train_list) // batch_size)
    epoch_times = []

    for epoch in range(num_epochs):
        np.random.shuffle(train_list)
        start_time = time.perf_counter()
        losses_over_epoch = 0
        for step in range(steps_per_epoch):
            images, labels = get_batch(train_list, batch_size, allow_augmentation=True)

            lr = get_lr(epoch, initial_lr, num_epochs)

            # forward pass
            out = images
            for lw in layers:
                out = lw.layer.forward(out)

            # loss
            loss = criterion.forward(out, labels)
            loss_values_per_iter.append(loss)
            losses_over_epoch += loss

            print("\r", end="")
            print(f"Epoch: {epoch + 1} Step: {step} | [{abs((step / steps_per_epoch) * 100):.1f} % complete] | Loss: {loss:.4f} | LR: {lr:.8f}",
                end="")

            # backward pass
            d_out = criterion.backward()

            for lw in reversed(layers):
                d_out = lw.layer.backward(d_out)

            # update weights with grad clipping
            for lw in layers:
                layer = lw.layer

                if isinstance(layer, Convolution):
                    clip_gradients(layer, max_norm=1.0)
                    layer.filters -= lr * (layer.dW + weight_decay_amount * layer.filters)
                    layer.bias -= lr * layer.db

                elif isinstance(layer, FullyConnected):
                    clip_gradients(layer, max_norm=1.0)
                    layer.weights -= lr * (layer.dW + weight_decay_amount * layer.weights)
                    layer.biases -= lr * layer.db

        # val
        end_time = time.perf_counter()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        time_remaining = (num_epochs - epoch - 1) * avg_epoch_time

        hours = int(time_remaining // 3600)
        minutes = int((time_remaining % 3600) // 60)

        aveLoss = losses_over_epoch / (steps_per_epoch + 0.0001)

        if (epoch+1) % 5 == 0 or epoch == 0:
            val_acc = evaluate(layers, test_list, directory_test, num_samples=batch_size * 10)
            val_acc = val_acc.get() if isinstance(val_acc, cp.ndarray) else val_acc

            val_accuracy_per_epoch.append((epoch, val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if verbose:
                print("\r", end="")
                print("\n", "=" * 160)
                print(f"Epoch {epoch + 1}/{num_epochs} | Epoch duration: {epoch_duration:.2f}s | "
                      f"Estimated time remaining: {hours}h {minutes}m"
                      f" | Ave Loss: {aveLoss:.4f} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f} ")
                print("=" * 160)
        else:
            print("\r", end="")
            print(f"\nEpoch {epoch + 1}/{num_epochs} | Epoch duration: {epoch_duration:.2f}s | "
                  f"Estimated time remaining: {hours}h {minutes}m | Ave Loss: {aveLoss:.4f}")

    return layers


# data loading
CIFAR10_class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
directory_train = r"CIFAR-10-images-master/train"
directory_test = r"CIFAR-10-images-master/test"

BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 30
WEIGHT_DECAY_AMOUNT = 1e-4

# % of training data
subset_fraction = 0.05


def scan_files_dataload_shuffled(directory):
    temp_list = []
    # counter = 0
    with os.scandir(directory) as es:
        for e in es:
            with os.scandir(e) as files:
                for file in files:
                    
                    temp_list.append((e, file))
    np.random.shuffle(temp_list)
    return temp_list


train_list = scan_files_dataload_shuffled(directory_train)
subset_size = int(len(train_list) * subset_fraction)
train_list = train_list[:subset_size]

test_list = scan_files_dataload_shuffled(directory_test)


def convert_to_pixel_values(imagePath):
    im = Image.open(imagePath)
    pix = im.load()
    smallerSize = min(im.size)

    pixel_tuples_in_arr = []
    for i in range(smallerSize):
        row = []
        for j in range(smallerSize):
            row.append(pix[i, j])
        pixel_tuples_in_arr.append(row)

    input_image_rgb = utilMethods.convert_rgb_image_to_cxhxw(pixel_tuples_in_arr)
    return np.array(input_image_rgb, dtype=np.float32)


def get_batch(target_list, batch_size, allow_augmentation):
    idx = np.random.choice(len(target_list), size=batch_size, replace=False)
    batch_items = [target_list[i] for i in idx]

    images = []
    labels = []

    for e, f in batch_items:
        image_path = f"{directory_train}/{e.name}/{f.name}"
        x = convert_to_pixel_values(image_path).astype(np.float32)
        
        
        if x.shape[0] == 3:  
            
            x = np.transpose(x, (1, 2, 0))

        if(allow_augmentation):
            pad = 4
            x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

            H, W, _ = x.shape
            top = np.random.randint(0, H - 32 + 1)
            left = np.random.randint(0, W - 32 + 1)
            x = x[top:top + 32, left:left + 32]

            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]

        x = (x / 255.0 - 0.5) / 0.5
        
        x = np.transpose(x, (2, 0, 1)) 

        images.append(x)
        labels.append(CIFAR10_class_labels.index(e.name))

    images = cp.array(images, dtype=cp.float32)
    labels = cp.array(labels, dtype=cp.int64)
    return images, labels


def my_get_batch_adapter(batch_size):
    return get_batch(train_list, batch_size)

# ----------------------------------------
print("=" * 80)
print("Starting Training...")
print(f"Batch size: {BATCH_SIZE}, Initial LR: {LEARNING_RATE} Weight Decay Amount: {WEIGHT_DECAY_AMOUNT} Epochs: {NUM_EPOCHS} (Training on {int(subset_fraction * 100)}% of dataset)")
print("=" * 80)

trained_layers = run_nn_train(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    initial_lr=LEARNING_RATE,
    weight_decay_amount=WEIGHT_DECAY_AMOUNT,
    get_batch_fn=my_get_batch_adapter,
    test_list=test_list,
    directory_test=directory_test,
    verbose=True
)

# plotting part
import matplotlib.pyplot as plt

loss_values = np.array([x.get() if isinstance(x, cp.ndarray) else x for x in loss_values_per_iter])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# plot loss
ax1.plot(loss_values, alpha=0.3, color='blue', label='Raw Loss')
window_size = 10
if len(loss_values) > window_size:
    cumsum_vec = np.cumsum(np.insert(loss_values, 0, 0))
    smoothed_loss = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    ax1.plot(range(window_size - 1, len(loss_values)), smoothed_loss,
             color='red', linewidth=2, label=f'Smoothed (window={window_size})')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Curve')
ax1.grid(True, alpha=0.3)
ax1.legend()

# plot val accuracy
if val_accuracy_per_epoch:
    epochs, accs = zip(*val_accuracy_per_epoch)
    ax2.plot(epochs, accs, marker='o', color='green', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.3, label='Random (10%)')
    ax2.legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
if val_accuracy_per_epoch:
    final_acc = val_accuracy_per_epoch[-1][1]
    best_acc = max(acc for _, acc in val_accuracy_per_epoch)
    print(f"Final validation accuracy: {final_acc:.2%}")
    print(f"Best validation accuracy:  {best_acc:.2%}")
print("=" * 80)

