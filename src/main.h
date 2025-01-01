#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define SIZE 2

#define TRAIN_RATIO 0.8
#define VAL_RATIO 0.1

#define LAMBDA 0.001 

#define INPUT_SIZE 2
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 1
#define MAX_SAMPLES 100

typedef struct {
    double weights[2];
    double bias;
} Neuron;

typedef struct {
    double w1[INPUT_SIZE * HIDDEN_SIZE]; 
    double w2[HIDDEN_SIZE * OUTPUT_SIZE]; 
    double b1[HIDDEN_SIZE]; 
    double b2[OUTPUT_SIZE]; 
} NeuralNetwork;

static void matmul(double* m1, double* m2, double* result, int rows, int cols);
static double sigmoid(double x);
static double sigmoidDeriv(double x);
static double MSELoss(int y_true[], double y_pred[], int samples);
static double feedforward(NeuralNetwork* n, double x[], double* h);
static void initializeWeights(double *matrix, int size);
static void initializeBiases(double *biases, int size);
static void train(NeuralNetwork* n, double train_data[][INPUT_SIZE], int train_labels[], double val_data[][INPUT_SIZE], int val_labels[], int train_size, int val_size, int epochs);
