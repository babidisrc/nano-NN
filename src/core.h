#ifndef __CORE_H_
#define __CORE_H_

#include <string.h>
#include <time.h>
#include <math.h>

#include "dataset.h"

#define LAMBDA 0.001 
#define EPOCHS 1000

#define INPUT_SIZE 784 // 28x28 pixels
#define OUTPUT_SIZE 10 // number of classes (0-9)

#define HIDDEN_SIZE_1 16 // number of neurons in the first hidden layer
#define HIDDEN_SIZE_2 16 // number of neurons in the second hidden layer

typedef struct {
    // weights and biases of the first hidden layer
    double **w1;        // input layer weights for first hidden layer
    double *b1;        // first hidden layer biases

    // weights and biases of the second hidden layer
    double **w2;               // weights from the first hidden layer to the second hidden layer
    double *b2;               // second hidden layer biases

    // output layer weights and biases
    double **w3;               // weights from the second hidden layer to the output layer
    double *b3;               // output layer biases
} NeuralNetwork;

double crossEntropy(double y_true[], double y_pred[], int classes);
double leakyRelu(double x);
double leakyReluDeriv(double x);
void softmax(double x[], int size);
void matmul(double **m1, double *m2, double *result, int rows, int cols);

void forwardPropagation(double *pixels, NeuralNetwork *n, double probs[], double h1[], double h2[]);
void trainModel(NeuralNetwork *n, Dataset *train, Dataset *val, int epochs);

void initializeNeuralNetwork(NeuralNetwork *n);
void freeNeuralNetwork(NeuralNetwork *n);

#endif