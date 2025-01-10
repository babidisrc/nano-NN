# nano-NN
A Simple Neural Network in C that can identify digits using the MNIST Dataset.

The MNIST dataset is a collection of 70,000 handwritten digits (0-9), split into 60,000 training images and 10,000 test images. Each image is a 28x28 grayscale image, making it a popular dataset for benchmarking machine learning models.

## Prerequisites
Before building nano-NN, ensure you have the following installed:
- GCC (or another C compiler)
- CMake (version 3.10 or higher)

# Build from Source

1. Clone the repository:

```bash
git clone https://github.com/babidisrc/nano-NN.git
cd nano-NN
```

2. Build the executable:

```bash
mkdir build
cd build
cmake ..
make
```

3. You are now good to go!

# Usage

```bash
Usage: ./nano-nn <imagefile>
```

- `<imagefile>`: Image with 28x28 pixels or resizable.

## Examples:
To test the neural network, you can use any 28x28 grayscale image of a handwritten digit. If your image is not 28x28, the program will resize it automatically.

**WARNING**
1. **Image Quality:** Ensure the digit is clearly visible and centered in the image. Poor-quality images or digits that are too small may lead to incorrect predictions.
2. **Aspect Ratio:** If the original image has a different aspect ratio (e.g., rectangular instead of square), the resizing process may distort the digit, making it harder for the neural network to recognize it.
3. **Image Size:** While the program can handle larger images, extremely large images may lose important details during resizing, making the digit unrecognizable. For best results, use images that are close to 28x28 pixels or have a similar aspect ratio.

Example command:

```bash
./nano-nn sample.jpg
```

Output example:

```bash
...
Epoch 15 - ValLoss: 0.1888, ValAccuracy: 94.54%
Epoch 16 - ValLoss: 0.1886, ValAccuracy: 94.57%
Epoch 17 - ValLoss: 0.1890, ValAccuracy: 94.64%
Epoch 18 - ValLoss: 0.1891, ValAccuracy: 94.63%
Epoch 19 - ValLoss: 0.1887, ValAccuracy: 94.60%
Epoch 20 - ValLoss: 0.1897, ValAccuracy: 94.53%

0 --> 0.999895%
1 --> 0.000000%
2 --> 0.000015%
3 --> 0.000001%
4 --> 0.000000%
5 --> 0.000050%
6 --> 0.000002%
7 --> 0.000000%
8 --> 0.000010%
9 --> 0.000026%

The digit is probably 0 with 99.99% confidence.
```
