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
Usage: ./nano-nn <dataset> [input number (n)] [input column 1...n] epochs
```

- `<dataset>`: Input file from any text extension (must have elements separated by commas).
- `[input number (n)]`: Number of inputs. Always an int.
- `[input column name 1..n]`: Columns name that will serve as an input.
- `epochs`: Number of training epochs. Always an int (100, 1, 1000, 250...)

Output/Label is always the last column for now.

## Examples:

Given the file `test.csv`:

```bash
Name,Weight,Height,Gender
Sophie,1,0,1
Leon,15,-1,0
Julia,-7,-8,1
Elias,20,0,0
...
Noah,13,2,0
Layla,-4,-5,1
Zachary,6,0,0
```

Run the following command:

```bash
./nano-nn test.csv 2 Weight Height 1000
```

Output example:

```bash
Epoch 0 - ValLoss: 0.2060
Epoch 10 - ValLoss: 0.0383
Epoch 20 - ValLoss: 0.0210
Epoch 30 - ValLoss: 0.0160
...
Input: {13.0, 2.0} -> Prediction: 0.072 (Expected: 0)
Input: {-4.0, -5.0} -> Prediction: 0.933 (Expected: 1)
Input: {6.0, 0.0} -> Prediction: 0.171 (Expected: 0)
```