/*
    hi! if you're here, you're probably interested in finding out how this code works. i'll try to explain it briefly. 
    for more detailed explanations, check out these resources:

    1: https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw
    2: https://www.youtube.com/watch?v=VMj-3S1tku0
    3: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
    4: https://www.youtube.com/watch?v=w8yWXqWQYmU&t=50s
    5: The Hundred-Page Machine Learning Book, by Andriy Burkov

    ---------------------

    a neural network is composed of neurons - they take inputs, does some pretty magic math with them, and produces one output:

    x1                             
    | \               
    x2  ---> *w -> +b -> f() --> y
    | /               
    x3              
    
    the example above is a neuron, the basic unit of a neural network:
    it has an input layer, with each input (x1, x2 and x3) being multiplied by a weight (w)

    x1 ​→ x1​ ∗ w1​ 
    x2 → x2 ∗ w2
    x3 → x3 ​∗ w3

    next, all the weighteds inputs are added together with a bias (b)

    (x1 ​∗ w1​) + (x2​ ∗ w2​) + (x3 ∗ w3) + b

    then the sum is passed through an activation function, generating our output (y)

    y = f((x1​ ∗ w1​) + (x2​ ∗ w2​) + (x3 ∗ w3) + b)

    this activation function is important to turn the output in a nice predictable form. 
    the most simple activation function (and that is used in this code) is the sigmoid function

    the sigmoid function "squish" values into a range between [0,1]
    it's way easier to work with this range than hyper high ranges like [-infinity, infinity], for example

    a neural network is a combination of multiple neurons:

    Input Layer       Hidden Layer     Output Layer
    x1                 h1                
    | \                | \               
    x2 --- W1 --->     h2 --- W2 --->    o1
    | /                | /               
    x3                 h3                 

    this network has 3 inputs, a hidden layer with 3 neurons (h1​, h2 and h3​), and an output layer with 1 neuron (o1​).

    a hidden layer is an intermediate layer between the input layer and the output layer. 
    you can have multiple hidden layers if you want

    so, this is the basics of neural networks, and knowing that the code will become a little easier to read (i hope).
*/

#include "core.h"

// activation function - range [0,1]
static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// sigmoid derivative function for backpropagation
static double sigmoidDeriv(double x) {
    double fx = sigmoid(x);
    return fx * (1 - fx);
}

// MSE: measures the performance of the model
// y_true[] = true label, the expected output (e.g. 0)
// y_pred[] = the model prediction (e.g. 1)

// if there is just 1 input (0) and it's wrong, the MSE would be 1 because:
// error = 0 - 1 = -1
// mean += (-1) * (-1) = 1
// return mean / samples --> return 1 / 1 -> return 1
static double meanSquaredError(int y_true[], double y_pred[], int samples) {
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
        result[i] = 0;
        for (int j = 0; j < cols; j++) {
            result[i] += m1[i * cols + j] * m2[j];
        }
    }
}

// forward propagation: calculates the output of the neural network from the inputs.
// it transforms the inputs (x) into an output (o) passing through weights, biases and activation functions.
static double forwardPropagation(NeuralNetwork* n, double x[], double* h, int input_size) {

    // multiplies the hidden layer weight matrix (n->w1) by the input vector (x),
    // resulting in the vector h which represents the inputs to the hidden layer neurons.
    matmul(n->w1, x, h, HIDDEN_SIZE, input_size);

    // h[i] = hidden layer neuron
    // e.g, if HIDDEN_SIZE = 5, then there's just 5 neurons
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        // sums progressively with the bias and "squish" the number to a range of [0,1]
        h[i] += n->b1[i];
        h[i] = sigmoid(h[i]);
    }

    // o = output
    // does the same thing as above but with the w2 and b2
    double o = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        o += n->w2[i] * (h[i]);
    }
    o += n->b2;

    // the weights (n->w1, n->w2) and biases (n->b1, n->b2) are parameters adjusted during training 
    // to minimize neural network error

    // then, it returns a value in the range [0, 1], due to the use of the sigmoid function
    return sigmoid(o);
}

// initialize weights randomly in a interval of [-1, 1]
static void initializeRandomWeights(double* weigths, int size) {
    for (int i = 0; i < size; i++) {
        weigths[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

// initialize biases randomly in a interval of [-1, 1]
static void initializeRandomBiases(double* biases, int size) {
    for (int i = 0; i < size; i++) {
        biases[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

static void trainModel(NeuralNetwork* n, Dataset* train, Dataset* val, int epochs, int input_size) {
    // prev_val_loss: stores the previous loss in the validation set.
    // no_improve_count: counts how many epochs passed without improvement in validation loss.
    // max_no_improve: max number of epochs without improvements before stopping training (early stopping)
    // learn_rate: controls the size of weight adjustments.

    double prev_val_loss = 999999, no_improve_count = 0, max_no_improve = 6;
    double learn_rate = 0.1;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < train->size; i++) {

            double h[HIDDEN_SIZE];
            double o = forwardPropagation(n, train->data[i], h, input_size);

            // error is the difference of train->labels[1] (desired output) and o (our output)
            double error = train->labels[i] - o;

            // backpropagation + L2
            double d_o = error * sigmoidDeriv(o);

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                // d_h = gradient of the loss in relation to the hidden layer activations.
                double d_h = d_o * n->w2[j] * sigmoidDeriv(h[j]);

                // updates hidden layer weights (n->w1) using gradient and L2 regularization (LAMBDA * n->w1[j * input_size + k])
                for (int k = 0; k < input_size; k++) {
                    double regularization_term = LAMBDA * n->w1[j * input_size + k];
                    n->w1[j * input_size + k] += learn_rate * (d_h * train->data[i][k] - regularization_term);
                }

                // update biases of hidden layer
                n->b1[j] += learn_rate * d_h;

                // update weights of output layer (n->w2) using gradient and L2 regularization
                double regularization_term_w2 = LAMBDA * n->w2[j];
                n->w2[j] += learn_rate * (d_o * h[j] - regularization_term_w2);
            }

            // update bias of output layer
            n->b2 += learn_rate * d_o;
        }

        // calculates network predictions on the validation set.
        double val_preds[val->size];
        for (int i = 0; i < val->size; i++) {
            double h[HIDDEN_SIZE];
            val_preds[i] = forwardPropagation(n, val->data[i], h, input_size);
        }

        // calculates loss on the validation set
        double val_loss = meanSquaredError(val->labels, val_preds, val->size);

        // early stopping -> if validation loss does not improve for max_no_improve epochs, training stops
        if (val_loss < prev_val_loss) {
            prev_val_loss = val_loss;
            no_improve_count = 0;
        } else {
            no_improve_count++;
            if( no_improve_count >= max_no_improve) 
                break;
        }

        // every 10 epochs, print validation loss.
        if (epoch % 10 == 0) {
            printf("Epoch %d - ValLoss: %.4f\n", epoch, val_loss);
        }
    }
}

void printUsage() {
    printf("nano-nn: A Simple Neural Network in C created for learning purposes\n\n"
           "Usage: ./nano-nn <dataset> [input number (n)] [input column 1...n] epochs\n\n"
           "<dataset>: Input file from any text extension (must have elements separated by commas).\n"
           "[input number (n)]: Number of inputs. Always an int.\n"
           "[input column name 1..n]: Columns name that will serve as an input.\n"
           "epochs: Number of training epochs. Always an int (100, 1, 1000, 250...)\n\n"
           "Examples:\n"
           "\t./nano-nn 2 Weight Height Gender 1000\n"
           "\t./nano-nn 4 Sleep_Hours Study_Hours Socioeconomic_Score Attendance Grades 250\n\n"
           "More information: https://github.com/babidisrc/nano-NN/\n");
}

void print_prediction(double* inputs, int num_inputs, double prediction, int expected) {
    printf("Input: {");
    
    for (int i = 0; i < num_inputs; i++) {
        printf("%.1f", inputs[i]);
        if (i < num_inputs - 1) {
            printf(", ");
        }
    }

    printf("} -> Prediction: %.3f (Expected: %d)\n", prediction, expected);
}

int main(int argc, char* argv[]) {
    if (argc == 2 && strcmp(argv[1], "-h") == 0) {
            printUsage();
            exit(EXIT_SUCCESS);
    }
    
    if (atoi(argv[2]) <= 0) {
        printf("Invalid input number (n). Must be a positive integer.\n");
        exit(EXIT_FAILURE);
    }

    if (argc != (atoi(argv[2]) + 4)) {
        printf("Usage: ./nano-nn <dataset> [input number (n)] [input column 1...n] epochs\n");
        exit(EXIT_FAILURE);
    }

    FILE* stream = fopen(argv[1], "r");
    if (!stream) {
        printf("Failed to open file: %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }

    char line[2048];
    int epochs = atoi(argv[argc - 1]);

    int input_size = atoi(argv[2]);

    char **input_name = malloc(input_size * sizeof(char *));
    if (!input_name) {
        printf("Memory allocation failed for input_name.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < input_size; i++) {
        input_name[i] = malloc(strlen(argv[3 + i]) + 1); // +1 -> '\0'
        strcpy(input_name[i], argv[3 + i]);
    }

    int input_columns[input_size];
    int input_count = 0;

    // data from the .csv file
    double data[MAX_SAMPLES][MAX_HEADERS] = {0};
    char headers[MAX_HEADERS][1024] = {0};
    
    int rows = 0, columns = 0, max_columns = 0;

    while (fgets(line, sizeof(line), stream)) {
        columns = 0;

        char* token = strtok(line, ","); 

        while (token) {
            if (rows == 0) {
                strcpy(headers[columns], token);
                
                for (int k = 0; k < input_size; k++) {
                    if (strcmp(input_name[k], headers[columns]) == 0) {
                        input_columns[input_count++] = columns;
                    
                    }
                }
                
            } else if (rows > 0) {  
                for (int k = 0; k < input_count; k++) {
                    if (columns == input_columns[k]) {
                        data[rows - 1][k] = strtod(token, NULL);
                        // printf("DATA[%d][%d]: %.2f\n", rows - 1, k, data[rows - 1][k]);
                    }
                }

                if (columns == max_columns - 1) {
                    data[rows - 1][input_count] = strtod(token, NULL);
                    // printf("Output[%d]: %.2f\n", i - 1, data[i - 1][input_count]);
                }
                
            }

            token = strtok(NULL, ","); 
            columns++;
            max_columns = fmax(columns, max_columns);
        }
        rows++;
    }  

    fclose(stream);

    int samples = rows;

    Dataset* train = createDataset((int)(samples * TRAIN_RATIO), input_size);
    Dataset* val = createDataset((int)(samples * VAL_RATIO), input_size);
    Dataset* test = createDataset((int)(samples - (train->size + val->size)), input_size);

    // expected output
    int all_y_trues[samples];
    
    splitDataset(&train, &val, &test, samples, input_size, data, all_y_trues);

    NeuralNetwork n;

    // random values for weights and biases
    srand(time(NULL));
    initializeRandomWeights(n.w1, input_size * HIDDEN_SIZE); 
    initializeRandomWeights(n.w2, HIDDEN_SIZE * OUTPUT_SIZE);  
    initializeRandomBiases(n.b1, HIDDEN_SIZE);
    initializeRandomBiases(&n.b2, OUTPUT_SIZE);

    // train neural network
    trainModel(&n, train, val, epochs, input_size);

    // evaluate predictions and output the results
    int correct = 0;

    printf("\nPredictions:\n");

    for (int i = 0; i < val->size; i++) {
        double h[HIDDEN_SIZE];
        double prediction = forwardPropagation(&n, test->data[i], h, input_size);

        print_prediction(test->data[i], input_size, prediction, test->labels[i]);
        
        int rounded_prediction = prediction >= 0.5 ? 1 : 0;

        if (rounded_prediction == test->labels[i]) {
            correct++;
        }

    }

    double accuracy = ((double)correct / (test->size - input_size)) * 100;;
    printf("\nTest Precision: %.2f%%\n", accuracy);

    // free all datasets and input_names
    freeDataset(train);
    freeDataset(val);
    freeDataset(test);

    for (int i = 0; i < input_size; i++) {
        free(input_name[i]);
    }

    return 0;
}
