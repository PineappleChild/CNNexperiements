# CNNexperiements

A from-scratch implementation of convolutional neural networks in Python — no PyTorch, no NumPy. Built to understand what actually happens inside a CNN before letting a framework do it for you.

---

## What this is

Most CNN tutorials have you calling `nn.Conv2d` and calling it a day. This repo builds the pieces manually: convolution, ReLU, max pooling, fully connected layers, and softmax — then uses that foundation to replicate VGGNet for image classification on CIFAR-10, accelerated with CuPy.

The goal was never to beat a framework on performance. It was to make sure I actually understood the mechanics before abstracting them away.

---

## Architecture

### Custom CNN (from scratch)
```
input (RGB, depth=3)
  -> (conv -> relu -> maxpool) x2       # feature extraction blocks
  -> flatten
  -> fc(120) -> fc(84) -> fc(10)        # classifier head
  -> softmax -> logits
```
All layers — convolution, backprop, pooling — implemented without NumPy or PyTorch.

### VGGNet Replication (CuPy)
```
# (conv -> bn -> relu) x2 -> mp -> dropout ->
# (conv -> bn -> relu) x2 -> mp -> dropout ->
# (conv -> bn -> relu) x3 -> mp -> dropout ->
# (conv -> bn -> relu) x2 -> mp -> dropout ->
# gap -> fc -> relu -> dropout -> fc -> logits
```
Built on top of the same custom layer foundation, with CuPy for GPU acceleration. Trained on CIFAR-10.

---

## Implementation details

- **Convolution**: manual kernel sliding, configurable stride and same-padding via `compute_padding()`
- **Pooling**: max pooling with configurable kernel size and stride
- **Activations**: ReLU applied post-convolution
- **Fully connected**: standard linear layers with manual weight init
- **Softmax**: applied at output for class probabilities
- **GPU support**: CuPy used in the VGGNet replication for accelerated computation

---

## Stack

- Python
- CuPy (VGGNet replication)
- No PyTorch / TensorFlow / NumPy for core layer logic
