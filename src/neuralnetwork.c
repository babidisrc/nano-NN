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
    one simple activation function is the ReLU function

    the ReLU function has a cool feature: 
    if the number is positive, it doesn't change; but if the number is negative, the number is now equal 0

    x = 5 ---> ReLU(5) = 5 [no change]
    x = -10 ---> ReLU(-10) = 0

    it's way easier to work this way cause it limits our range

    a neural network is a combination of multiple neurons:

    Input Layer       Hidden Layer     Output Layer
    x1                 h1                
    | \                | \               
    x2 --- W1 --->     h2 --- W2 --->    o1
    | /                | /               
    x3                 h3                 

    the network above has 3 inputs, a hidden layer with 3 neurons (h1​, h2 and h3​), and an output layer with 1 neuron (o1​).

    a hidden layer is an intermediate layer between the input layer and the output layer. 
    you can have multiple hidden layers if you want

    so, this is the basics of neural networks, and knowing that the code will become a little easier to read (i hope).
*/

#include "core.h"

// initialize neural networks
void initializeNeuralNetwork(NeuralNetwork *n) {

    n->w1 = malloc(HIDDEN_SIZE_1 * sizeof(double *));
    if (!n->w1) fatal("Failed to allocate memory");
    
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        n->w1[i] = malloc(INPUT_SIZE * sizeof(double));
        if (!n->w1[i]) fatal("Failed to allocate memory");
        
        for (int j = 0; j < INPUT_SIZE; j++) {
            n->w1[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / INPUT_SIZE); // He initialization
        }
    }

    n->w2 = malloc(HIDDEN_SIZE_2 * sizeof(double *));
    if (!n->w2) fatal("Failed to allocate memory");
    
    for (int i = 0; i < HIDDEN_SIZE_2; i++) {
        n->w2[i] = malloc(HIDDEN_SIZE_1 * sizeof(double));
        if (!n->w2[i]) fatal("Failed to allocate memory");
        
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {     
            n->w2[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / HIDDEN_SIZE_1); // He initialization
        }
    }

    n->w3 = malloc(OUTPUT_SIZE * sizeof(double *));
    if (!n->w3) fatal("Failed to allocate memory");
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        n->w3[i] = malloc(HIDDEN_SIZE_2 * sizeof(double));
        if (!n->w3[i]) fatal("Failed to allocate memory");
        
        for (int j = 0; j < HIDDEN_SIZE_2; j++) {
            n->w3[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / HIDDEN_SIZE_2); // He initialization
        }
    }

    n->b1 = malloc(HIDDEN_SIZE_1 * sizeof(double));
    n->b2 = malloc(HIDDEN_SIZE_2 * sizeof(double));
    n->b3 = malloc(OUTPUT_SIZE * sizeof(double));
    
    if (!n->b1 || !n->b2 || !n->b3) fatal("Failed to allocate memory");

    for (int i = 0; i < HIDDEN_SIZE_1; i++) n->b1[i] = 0.0;
    for (int i = 0; i < HIDDEN_SIZE_2; i++) n->b2[i] = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) n->b3[i] = 0.0;
}

// free all the memory used
void freeNeuralNetwork(NeuralNetwork *n) {
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        free(n->w1[i]);
    }
    free(n->w1);
    free(n->b1);

    for (int i = 0; i < HIDDEN_SIZE_2; i++) {
        free(n->w2[i]);
    }
    free(n->w2);
    free(n->b2);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        free(n->w3[i]);
    }
    free(n->w3);
    free(n->b3);
}

// using leaky ReLU to avoid dead neurons (neurons that always returns 0)
double leakyRelu(double x) {
    return x > 0 ? x : 0.01 * x; // Leaky ReLU, alpha = 0.01
}

double leakyReluDeriv(double x) {
    return x > 0 ? 1 : 0.01; // Deriv Leaky ReLU
}

// measures the difference between the predicted and actual probability distribution

// y_true[i] = true label, the expected output
// y_pred[i] = the model prediction
double crossEntropy(double y_true[], double y_pred[], int classes) {
    double result = 0;
    const double epsilon = 1e-15; 

    for (int i = 0; i < classes; i++) {
        // y_pred[i] in range [0,1]
        double prob = y_pred[i];
        if (prob < epsilon) prob = epsilon; // avoid log(0)
        if (prob > 1 - epsilon) prob = 1 - epsilon; // avoid log(1)

        double error = y_true[i] * log(prob + epsilon);
        result += error;
    }

    return -result;
}

// matrix multiplication
void matmul(double **m1, double *m2, double *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0;
        for (int j = 0; j < cols; j++) {
            result[i] += m1[i][j] * m2[j];
        }
    }
}

// forward propagation: calculates the output of the neural network from the inputs
// it transforms the inputs (x) into an output passing through weights, biases and activation functions
void forwardPropagation(double *pixels, NeuralNetwork *n, double probs[], double h1[], double h2[]) {
    int i, j;

    // hidden layer 1
    matmul(n->w1, pixels, h1, HIDDEN_SIZE_1, INPUT_SIZE);

    for (i = 0; i < HIDDEN_SIZE_1; i++) {
        h1[i] += n->b1[i];
        h1[i] = leakyRelu(h1[i]); // activation function
    }

    // hidden layer 2
    matmul(n->w2, h1, h2, HIDDEN_SIZE_2, HIDDEN_SIZE_1);

    for (i = 0; i < HIDDEN_SIZE_2; i++) {
        h2[i] += n->b2[i];
        h2[i] = leakyRelu(h2[i]); // activation function
    }

    // output layer
    double output[OUTPUT_SIZE] = {0};
    matmul(n->w3, h2, output, OUTPUT_SIZE, HIDDEN_SIZE_2);

    for (i = 0; i < OUTPUT_SIZE; i++) {
        output[i] += n->b3[i];
    }

    // apply softmax to get probabilities
    softmax(output, OUTPUT_SIZE);

    // copy results to probs
    for (i = 0; i < OUTPUT_SIZE; i++) {
        probs[i] = output[i];
    }
}

// softmax: "squashes" a vector of size K between [0,1]]
// because it is a normalization of the exponential, the sum of this whole vector equates to 1.
// output of the softmax is the probabilities that a certain set of features belongs to a certain class
// example output:

// 0.999984
// 0.000003
// 0.000007
// 0.000002
// 0.000004

// notice how the sum of all theses probabilities is equal to 1
 void softmax(double activations[], int size) {
    double sum = 0;
    double max_x = activations[0];

    for (int i = 1; i < size; i++) {
        if (activations[i] > max_x) {
            max_x = activations[i];
        }
    }

    for (int i = 0; i < size; i++) {
        activations[i] = exp(activations[i] - max_x); // avoid overflow
        sum += activations[i];
    }

    // normalize to obtain the probabilities
    for (int i = 0; i < size; i++) {
        activations[i] /= sum;
    }
    
}

// trains the neural network using the provided training and validation datasets.
void trainModel(NeuralNetwork *n, Dataset *train, Dataset *val, int epochs) {
    // prev_val_loss: stores the previous loss in the validation set.
    // no_improve_count: counts how many epochs passed without improvement in validation loss.
    // max_no_improve: max number of epochs without improvements before stopping training (early stopping)
    // learn_rate: controls the size of weight adjustments.

    double prev_val_loss = 999999, no_improve_count = 0, max_no_improve = 10;
    double learn_rate = 0.001;

    double probs[OUTPUT_SIZE];

    double h1[HIDDEN_SIZE_1] = {0};
    double h2[HIDDEN_SIZE_2] = {0};

    double normalized_pixels[INPUT_SIZE] = {0};

    Dataset batch;
    int batches = (train->numfiles_images + BATCH_SIZE - 1) / BATCH_SIZE; 

    for (int i = 0; i < epochs; i++) {
        for (int b = 0; b < batches; b++) {
            // create a new batch
            if (createBatch(train, &batch, BATCH_SIZE, b) != 1) {
                fatal("Couldn't create batch dataset.");
            }

            for (int j = 0; j < batch.numfiles_images; j++) {
                // normalize pixel values to the range [0, 1] using PIXEL_SCALE.
                for (int k = 0; k < INPUT_SIZE; k++) {
                    normalized_pixels[k] = PIXEL_SCALE(batch.pixels[j * INPUT_SIZE + k]);
                }

                forwardPropagation(normalized_pixels, n, probs, h1, h2);

                // one-hot encoding (e.g., label 2 becomes [0, 0, 1, 0, ..., 0])
                double y_true[OUTPUT_SIZE] = {0};
                y_true[batch.labels[j]] = 1;

                // backpropagation

                // gradient output layer
                double d_output[OUTPUT_SIZE];
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    d_output[k] = probs[k] - y_true[k]; //  predicted output - true label
                }

                // gradient hidden layer 2
                double d_h2[HIDDEN_SIZE_2] = {0};
                for (int k = 0; k < HIDDEN_SIZE_2; k++) {
                    for (int l = 0; l < OUTPUT_SIZE; l++) {
                        d_h2[k] += d_output[l] * n->w3[l][k];
                    }
                    d_h2[k] *= leakyReluDeriv(h2[k]);
                }

                // gradient hidden layer 1
                double d_h1[HIDDEN_SIZE_1] = {0};
                for (int k = 0; k < HIDDEN_SIZE_1; k++) {
                    for (int l = 0; l < HIDDEN_SIZE_2; l++) {
                        d_h1[k] += d_h2[l] * n->w2[l][k];
                    }
                    d_h1[k] *= leakyReluDeriv(h1[k]);
                }

                // update weights and biases using gradient descent.
                // learn_rate controls the step size of the updates.
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    for (int l = 0; l < HIDDEN_SIZE_2; l++) {
                        n->w3[k][l] -= learn_rate * d_output[k] * h2[l];
                    }
                    n->b3[k] -= learn_rate * d_output[k];
                }

                for (int k = 0; k < HIDDEN_SIZE_2; k++) {
                    for (int l = 0; l < HIDDEN_SIZE_1; l++) {
                        n->w2[k][l] -= learn_rate * d_h2[k] * h1[l];
                    }
                    n->b2[k] -= learn_rate * d_h2[k];
                }

                for (int k = 0; k < HIDDEN_SIZE_1; k++) {
                    for (int l = 0; l < INPUT_SIZE; l++) {
                        n->w1[k][l] -= learn_rate * d_h1[k] * normalized_pixels[l];
                    }
                    n->b1[k] -= learn_rate * d_h1[k];
                }
            }
        }

        // calculate validation loss and accuracy
        double val_loss = 0;
        int correct_predictions = 0; // count of correctly predicted images

        for (int i = 0; i < val->numfiles_images; i++) {
            double h1[HIDDEN_SIZE_1] = {0};
            double h2[HIDDEN_SIZE_2] = {0};
            double probs[OUTPUT_SIZE];  

            for (int k = 0; k < INPUT_SIZE; k++) {
                normalized_pixels[k] = PIXEL_SCALE(val->pixels[i * INPUT_SIZE + k]);
            }

            forwardPropagation(normalized_pixels, n, probs, h1, h2);

            // convert label to one-hot encoding
            double y_true[OUTPUT_SIZE] = {0};
            y_true[val->labels[i]] = 1;

            // calculate cross-entropy loss
            val_loss += crossEntropy(y_true, probs, OUTPUT_SIZE);

            // determines the predicted digit
            int predicted_digit = 0;
            double max_prob = probs[0];
            for (int k = 1; k < OUTPUT_SIZE; k++) {
                if (probs[k] > max_prob) {
                    predicted_digit = k;
                    max_prob = probs[k];
                }
            }

            // compare with label
            if (predicted_digit == val->labels[i]) {
                correct_predictions++;
            }
        }

        val_loss /= val->numfiles_images;
        double accuracy = (double)correct_predictions / val->numfiles_images;

        // early stopping: stop training if validation loss doesn't improve for `max_no_improve` epochs
        if (val_loss < prev_val_loss) {
            prev_val_loss = val_loss;
            no_improve_count = 0;
        } else {
            no_improve_count++;
            if (no_improve_count >= max_no_improve) 
                break;
        }

        // print validation loss and accuracy for the current epoch.
        printf("Epoch %d - ValLoss: %.4f, ValAccuracy: %.2f%%\n", i, val_loss, accuracy * 100);

    }
}