#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define SIZE 2

#define INPUT_SIZE 2
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 1
#define SAMPLES 4

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
    double w1[INPUT_SIZE * HIDDEN_SIZE]; 
    double w2[HIDDEN_SIZE * OUTPUT_SIZE]; 
    double b1[HIDDEN_SIZE]; 
    double b2[OUTPUT_SIZE]; 
} NeuralNetwork;


void matmul(double* m1, double* m2, double* result, int rows, int cols);
double sigmoid(double x);
double sigmoidDeriv(double x);
double MSELoss(int y_true[], double y_pred[], int samples);
double feedforwardNetwork(NeuralNetwork* n, double x[]);
void initializeWeights(double *matrix, int rows, int cols);
void initializeBiases(double *biases, int size);
void train(NeuralNetwork* n, double data[][2], int all_y_trues[], int samples);
const char* getfield(char* line, int num);
