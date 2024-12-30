#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SIZE 2

typedef struct {
    double weights[2];
    double bias;
} Neuron;

// Neural Network with:
//     - 2 inputs
//     - a hidden layer with 2 neurons (h1, h2)
//     - an output layer with 1 neuron (o1)
//   Each neuron has the same weights and bias
    
typedef struct {
    double weights[2]; 
    double bias;
    Neuron h1;
    Neuron h2;
    Neuron o1;       
} NeuralNetwork;

void mat(double m1[], double m2[], double* mr)
{
    *mr = 0;
    for (int i = 0; i < 2; i++) {
        *mr += m1[i] * m2[i];
    }
}

double sigmoid(double x) {
    // sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + exp(-x)); 
}

double sigmoidDeriv(double x) {
    // derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    double fx = sigmoid(x);
    return fx * (1 - fx);
}

double MSELoss(int y_true[], int y_pred[], int samples){
    int result[samples];
    double mean;

    for (int i = 0; i < samples; i++) {
        result[i] = y_true[i] - y_pred[i];
        result[i] = pow(result[i], 2);
    }

    for (int i = 0; i < samples; i++) {
        mean += result[i];
    }

    return mean / samples;
}

double feedforwardNeuron(Neuron* n, double x[]) {
    double result; 
    mat(n->weights, x, &result);
    double total = result + n->bias;

    return sigmoid(total);
}

double feedforwardNetwork(NeuralNetwork* n, double x[]) {
    double out_h1 = feedforwardNeuron(&n->h1, x);
    double out_h2 = feedforwardNeuron(&n->h2, x);

    double x2[SIZE] = {out_h1, out_h2};

    double out_o1 = feedforwardNeuron(&n->h2, x2);

    return out_o1;
}