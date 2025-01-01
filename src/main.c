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
        for (int j = 0; j < cols; j++) {
            result[i] += m1[i * cols + j] * m2[j];
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

static void initializeWeights(double* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

static void initializeBiases(double* biases, int size) {
    for (int i = 0; i < size; i++) {
        biases[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

static void train(NeuralNetwork* n, double train_data[][INPUT_SIZE], int train_labels[], double val_data[][INPUT_SIZE], int val_labels[], int train_size, int val_size, int epochs) {
    // regularization l2 + early stopping
    double prev_val_loss = 999999, no_improve_count = 0, max_no_improve = 6;
    double learn_rate = 0.1;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < train_size; i++) {

            // feedforward
            double h[HIDDEN_SIZE];
            double o = feedforward(n, train_data[i], h);

            double error = train_labels[i] - o;

            // backpropagation + L2
            double d_o = error * sigmoidDeriv(o);

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double d_h = d_o * n->w2[j] * sigmoidDeriv(h[j]);

                for (int k = 0; k < INPUT_SIZE; k++) {
                    double regularization_term = LAMBDA * n->w1[j * INPUT_SIZE + k];
                    n->w1[j * INPUT_SIZE + k] += learn_rate * (d_h * train_data[i][k] - regularization_term);
                }

                n->b1[j] += learn_rate * d_h;

                double regularization_term_w2 = LAMBDA * n->w2[j];
                n->w2[j] += learn_rate * (d_o * h[j] - regularization_term_w2);
            }

            n->b2[0] += learn_rate * d_o;
        }

        double val_preds[val_size];
        for(int i = 0; i < val_size; i++){
            double h[HIDDEN_SIZE];
            val_preds[i] = feedforward(n, val_data[i], h);
        }

        double val_loss = MSELoss(val_labels, val_preds, val_size);

        // Early stopping
        if (val_loss < prev_val_loss) {
            prev_val_loss = val_loss;
            no_improve_count = 0;
        } else {
            no_improve_count++;
            if( no_improve_count >= max_no_improve) 
                break;
        }

        if (epoch % 10 == 0) {
            printf("Epoch %d - ValLoss: %.4f\n", epoch, val_loss);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: ./nano-nn <dataset> epochs\n");
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
    double data[MAX_SAMPLES][MAX_SAMPLES] = {0};
    
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

    int samples = i;
    
    // divide in train, test and validation
    int train_size = (int)(samples * TRAIN_RATIO);
    int val_size = (int)(samples * VAL_RATIO);
    int test_size = samples - (train_size + val_size);

    double train_data[train_size][INPUT_SIZE];
    int train_labels[train_size];

    double val_data[val_size][INPUT_SIZE];
    int val_labels[val_size];

    double test_data[test_size][INPUT_SIZE];
    int test_labels[test_size];

    // expected output
    int all_y_trues[samples];
    
    for (int i = 0; i < samples; i++) {
        all_y_trues[i] = (int)data[i][INPUT_SIZE]; 
    }

    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            train_data[i][j] = data[i][j];
        }

        train_labels[i] = all_y_trues[i];
    }
    
    for (int i = 0; i < val_size; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
                val_data[i][j] = data[train_size + i][j];
        }

        val_labels[i] = all_y_trues[train_size + i];
    }

    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
                test_data[i][j] = data[train_size + val_size + i][j];
        }

        test_labels[i] = all_y_trues[train_size + val_size + i];
    }

    NeuralNetwork n;

    // random values for weights and biases
    srand(time(NULL));
    initializeWeights(n.w1, INPUT_SIZE * HIDDEN_SIZE); 
    initializeWeights(n.w2, HIDDEN_SIZE * OUTPUT_SIZE);  
    initializeBiases(n.b1, HIDDEN_SIZE);
    initializeBiases(n.b2, OUTPUT_SIZE);

    // train neural network
    train(&n, train_data, train_labels, val_data, val_labels, train_size, val_size, epochs);

    // evaluate predictions and output the results
    int correct = 0;

    printf("\nPredictions:\n");

    for (int i = 0; i < val_size; i++) {
        double h[HIDDEN_SIZE];
        double prediction = feedforward(&n, test_data[i], h);

        printf("Input: {%.1f, %.1f} -> Prediction: %.3f (Expected: %d)\n", 
                test_data[i][0], test_data[i][1], prediction, test_labels[i]);

        int rounded_prediction = prediction >= 0.5 ? 1 : 0;

        if (rounded_prediction == test_labels[i]) {
            correct++;
        }
    }

    // accuracy in %
    double accuracy = (double)correct / test_size * 100;
    printf("\nFinal Precision: %.2f%%\n", accuracy);

    return 0;
}
