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

static void trainNetwork(NeuralNetwork* n, Dataset* train, Dataset* val, int epochs) {
    // regularization l2 + early stopping
    double prev_val_loss = 999999, no_improve_count = 0, max_no_improve = 6;
    double learn_rate = 0.1;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < train->size; i++) {

            // feedforward
            double h[HIDDEN_SIZE];
            double o = feedforward(n, train->data[i], h);

            double error = train->labels[i] - o;

            // backpropagation + L2
            double d_o = error * sigmoidDeriv(o);

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double d_h = d_o * n->w2[j] * sigmoidDeriv(h[j]);

                for (int k = 0; k < INPUT_SIZE; k++) {
                    double regularization_term = LAMBDA * n->w1[j * INPUT_SIZE + k];
                    n->w1[j * INPUT_SIZE + k] += learn_rate * (d_h * train->data[i][k] - regularization_term);
                }

                n->b1[j] += learn_rate * d_h;

                double regularization_term_w2 = LAMBDA * n->w2[j];
                n->w2[j] += learn_rate * (d_o * h[j] - regularization_term_w2);
            }

            n->b2[0] += learn_rate * d_o;
        }

        double val_preds[val->size];
        for(int i = 0; i < val->size; i++){
            double h[HIDDEN_SIZE];
            val_preds[i] = feedforward(n, val->data[i], h);
        }

        double val_loss = MSELoss(val->labels, val_preds, val->size);

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

// 
Dataset* createDataset(int size, int input_size) {
    Dataset* dataset = malloc(sizeof(Dataset));
    dataset->size = size;
    dataset->input_size = input_size;

    // flexible array member
    dataset->data = malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        dataset->data[i] = malloc(input_size * sizeof(double));
    }

    dataset->labels = malloc(size * sizeof(int));

    return dataset;
}

void freeDataset(Dataset* dataset) {
    for (int i = 0; i < dataset->size; i++) {
        free(dataset->data[i]);
    }
    free(dataset->data);
    free(dataset->labels);
    free(dataset);
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

    Dataset* train = createDataset((int)(samples * TRAIN_RATIO), INPUT_SIZE);
    Dataset* val = createDataset((int)(samples * VAL_RATIO), INPUT_SIZE);
    Dataset* test = createDataset((int)(samples - (train->size + val->size)), INPUT_SIZE);

    // expected output
    int all_y_trues[samples];
    
    for (int i = 0; i < samples; i++) {
        all_y_trues[i] = (int)data[i][INPUT_SIZE]; 
    }

    for (int i = 0; i < train->size; i++) {
        for (int j = 0; j < train->input_size; j++) {
            train->data[i][j] = data[i][j];
        }

        train->labels[i] = all_y_trues[i];
    }
    
    for (int i = 0; i < val->size; i++) {
        for (int j = 0; j < val->input_size; j++) {
                val->data[i][j] = data[train->size + i][j];
        }

        val->labels[i] = all_y_trues[train->size + i];
    }

    for (int i = 0; i < test->size; i++) {
        for (int j = 0; j < test->input_size; j++) {
                test->data[i][j] = data[train->size + val->size + i][j];
        }

        test->labels[i] = all_y_trues[train->size + val->size + i];
    }

    NeuralNetwork n;

    // random values for weights and biases
    srand(time(NULL));
    initializeWeights(n.w1, INPUT_SIZE * HIDDEN_SIZE); 
    initializeWeights(n.w2, HIDDEN_SIZE * OUTPUT_SIZE);  
    initializeBiases(n.b1, HIDDEN_SIZE);
    initializeBiases(n.b2, OUTPUT_SIZE);

    // train neural network
    trainNetwork(&n, train, val, epochs);

    // evaluate predictions and output the results
    int correct = 0;

    printf("\nPredictions:\n");

    for (int i = 0; i < val->size; i++) {
        double h[HIDDEN_SIZE];
        double prediction = feedforward(&n, test->data[i], h);

        printf("Input: {%.1f, %.1f} -> Prediction: %.3f (Expected: %d)\n", 
                test->data[i][0], test->data[i][1], prediction, test->labels[i]);
        
        int rounded_prediction = prediction >= 0.5 ? 1 : 0;

        if (rounded_prediction == test->labels[i]) {
            correct++;
        }

    }

    double accuracy = ((double)correct / (test->size - INPUT_SIZE)) * 100;;
    printf("\nTest Precision: %.2f%%\n", accuracy);

    // free all datasets
    freeDataset(train);
    freeDataset(val);
    freeDataset(test);

    return 0;
}
