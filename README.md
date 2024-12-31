# nano-NN
A Simple Neural Network in C created for learning purposes, utilizing basic machine learning algorithms like backpropagation. It is not intended for production or development use.

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
./nano-nn <file> epochs
```

- `<file>`: Input file from any text extension (must have elements separated by commas).
- `<epochs>`: Number of training epochs. Always an int (100, 1, 100, 250...)

## Example:

Given the file `test.csv`:

```bash
Alice,-2,-1,1
Mina,-1,-6,1
Bob,25,6,0
Charlie,17,4,0
```

Run the following command:

```bash
./nano-nn test.csv 1000
```

Output example:

```bash
Epoch 10 Loss: 0.134
Epoch 20 Loss: 0.098
...
Input: {-2.0, -1.0} -> Prediction: 0.922 (Expected: 1)
Input: {-1.0, -6.0} -> Prediction: 0.939 (Expected: 1)
Input: {25.0, 6.0} -> Prediction: 0.012 (Expected: 0)
Input: {17.0, 4.0} -> Prediction: 0.019 (Expected: 0)
```

## TODO
- [X] Support for `.csv` files (WIP, only working with 4 columns for now)
- [ ] Faster implementation
- [ ] More helpful comments
- [ ] Expand functionality (e.g., dynamic number of columns)
- [ ] Maybe more...?
