# CNNexperiements

A from scratch implementation of CNN's in Python without PyTorch and NumPy for main CNN functionality.

---

## What this is

This repo builds the CNN manually: convolution, ReLU, max pooling, fully connected layers, softmax, etc. then uses that foundation to replicate VGGNet for image classification on CIFAR-10, accelerated with CuPy.

The goal was to understand the mechanics before abstracting them away.

---

## Architecture

### Custom CNN (from scratch)
```
-> (conv -> relu -> maxpool) x2       
-> flatten
-> fc -> fc -> fc        
-> softmax -> logits
```
All layers convolution, backprop, pooling, etc. implemented without NumPy or PyTorch.

### VGGNet Replication (CuPy)
```
(conv -> bn -> relu) x2 -> mp -> dropout ->
(conv -> bn -> relu) x2 -> mp -> dropout ->
(conv -> bn -> relu) x3 -> mp -> dropout ->
(conv -> bn -> relu) x2 -> mp -> dropout ->
gap -> fc -> relu -> dropout -> fc -> logits
```
Built on top of the same custom layer foundation with some improvements for speed, with CuPy for GPU acceleration, although I named the file using_numpy despite using cupy. 
Trained on CIFAR-10.

---

## Implementation details

- **Convolution**: manual kernel sliding, configurable stride and same-padding via `compute_padding()`
- **Pooling**: max pooling with configurable kernel size and stride
- **Activations**: ReLU applied post-convolution
- **Fully connected**: standard linear layers with manual weight init
- **Softmax**: applied at output for class probabilities
- **GPU support**: CuPy used in the VGGNet replication for accelerated computation
