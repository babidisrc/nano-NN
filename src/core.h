#ifndef __CORE_H_
#define __CORE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "parser.h"

#define SIZE 2

#define LAMBDA 0.001 

#define INPUT_SIZE 10
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 1

typedef struct {
    double w1[INPUT_SIZE * HIDDEN_SIZE]; 
    double w2[OUTPUT_SIZE * HIDDEN_SIZE]; 
    double b1[HIDDEN_SIZE]; 
    double b2; 
} NeuralNetwork;

static void matmul(double* m1, double* m2, double* result, int rows, int cols);
static double sigmoid(double x);
static double sigmoidDeriv(double x);
static double meanSquaredError(int y_true[], double y_pred[], int samples);
static double forwardPropagation(NeuralNetwork* n, double x[], double* h, int input_size);
static void trainModel(NeuralNetwork* n, Dataset* train, Dataset* val, int epochs, int input_size);

static void initializeRandomWeights(double *matrix, int size);
static void initializeRandomBiases(double *biases, int size);
static void freeBiasesWeights(double* biases, double* weights, int sizeB, int sizeW);

void print_prediction(double* inputs, int num_inputs, double prediction, int expected);

#endif