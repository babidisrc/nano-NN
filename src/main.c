#include "main.h"

static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

static double sigmoidDeriv(double x) {
    return x * (1 - x);
}

static double MSELoss(int y_true[], double y_pred[], int samples) {
    double mean = 0;

    for (int i = 0; i < samples; i++) {
        double error = y_true[i] - y_pred[i];
        mean += error * error;
    }

    return mean / samples;
}

// matrix multiplication
static void matmul(double* m1, double* m2, double* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            int row_offset = i * cols;
            result[i] += m1[row_offset + j] * m2[j];
        }
    }
}

static double feedforward(NeuralNetwork* n, double x[], double* h) {

    matmul(n->w1, x, h, HIDDEN_SIZE, INPUT_SIZE);

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h[i] += n->b1[i];
        h[i] = sigmoid(h[i]);
    }

    double o = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        o += n->w2[i] * (h[i]);
    }
    o += n->b2[0];

    return sigmoid(o);
}

static void initializeWeights(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

static void initializeBiases(double* biases, int size) {
    for (int i = 0; i < size; i++) {
        biases[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

static void train(NeuralNetwork* n, double data[][SAMPLES], int all_y_trues[], int samples, int epochs) {
    double learn_rate = 0.1; 
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < samples; i++) {
            int y_true = all_y_trues[i];

            // feedforward
            double h[HIDDEN_SIZE];
            double o = feedforward(n, data[i], h);

            double error = y_true - o;

            // backpropagation
            double d_o = error * sigmoidDeriv(o);

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double d_h = d_o * n->w2[j] * sigmoidDeriv(h[j]);
                for (int k = 0; k < INPUT_SIZE; k++) {
                    n->w1[j * INPUT_SIZE + k] += learn_rate * d_h * data[i][k];
                }
                n->b1[j] += learn_rate * d_h;
            }

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                n->w2[j] += learn_rate * d_o * h[j];
            }
            n->b2[0] += learn_rate * d_o;
        }

        if (epoch % 10 == 0) {
            double y_preds[SAMPLES];
            for (int i = 0; i < samples; i++) {
                // calculate y_preds applying feedforward in each line
                double h[HIDDEN_SIZE];
                y_preds[i] = feedforward(n, data[i], h);
            }

            double loss = MSELoss(all_y_trues, y_preds, samples);
            printf("Epoch %d loss: %.3f\n", epoch, loss);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: ./nano-nn <file> epochs\n");
        exit(EXIT_FAILURE);
    }

    FILE* stream = fopen(argv[1], "r");
    if (!stream) {
        printf("Failed to open file: %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }

    char line[1024];
    int epochs = atoi(argv[2]);

    // data from the .csv file
    double data[SAMPLES][SAMPLES] = {0};
    
    int i = 0, j = 0;

    while (fgets(line, sizeof(line), stream)) {
        j = 0;

        char* token = strtok(line, ","); 

        while (token) {
            if (j > 0) {  
                data[i][j - 1] = strtod(token, NULL); 
                // printf("DATA[%d][%d]: %.2f\n", i, j - 1, data[i][j - 1]);
            }

            token = strtok(NULL, ","); 
            j++;
        }
        i++;
    }  

    fclose(stream);

    // expected output
    int all_y_trues[SAMPLES];
    int aux = 0;

    for (int i = 0; i < SAMPLES; i++) {
        all_y_trues[i] = (int)data[i][INPUT_SIZE];
        if(data[i][0] == 0 && data[i][1] == 0)
        {
            aux = i;
            break;
        }
    }

    NeuralNetwork n;

    // random values for weights and biases
    srand(time(NULL));
    initializeWeights(n.w1, INPUT_SIZE, HIDDEN_SIZE);
    initializeWeights(n.w2, HIDDEN_SIZE, OUTPUT_SIZE);
    initializeBiases(n.b1, HIDDEN_SIZE);
    initializeBiases(n.b2, OUTPUT_SIZE);

    // train neural network
    train(&n, data, all_y_trues, aux, epochs);

    // evaluate predictions and output the results
    int correct = 0;

    printf("\nPredictions:\n");

    for (int i = 0; i < aux; i++) {
        double h[HIDDEN_SIZE];
        double prediction = feedforward(&n, data[i], h);

        printf("Input: {%.1f, %.1f} -> Prediction: %.3f (Expected: %d)\n", 
                data[i][0], data[i][1], prediction, all_y_trues[i]);

        int rounded_prediction = prediction >= 0.5 ? 1 : 0;

        if (rounded_prediction == all_y_trues[i]) {
            correct++;
        }
    }

    // accuracy in %
    double accuracy = (double)correct / aux * 100;
    printf("\nFinal Precision: %.2f%%\n", accuracy);

    return 0;
}
