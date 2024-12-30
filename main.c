#include "neuron.h"

int main(int argc, char** argv) {
    double weights[SIZE] = {0, 1};
    double bias = 0;
    double x[SIZE] = {2, 3};

    NeuralNetwork n = {.bias = bias};
    n.h1.bias = n.h2.bias = n.o1.bias = bias;

    memcpy(n.weights, weights, sizeof(double) * SIZE);
    memcpy(n.h1.weights, weights, sizeof(double) * SIZE);
    memcpy(n.h2.weights, weights, sizeof(double) * SIZE);
    memcpy(n.o1.weights, weights, sizeof(double) * SIZE);

    printf("%f\n", feedforwardNetwork(&n, x));
    return 0;
}