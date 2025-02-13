#include "core.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

void loadImage(const char *filename, double image[]) { 
    int width, height, channels;
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 1);
    if (!data) {
        fatal("Couldn't open image file");
    }

    stbir_resize_uint8_linear(data, width, height, 0, (unsigned char *)image, IMAGE_WIDTH, IMAGE_HEIGHT, 0, 1);

    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++) {
        image[i] = PIXEL_SCALE(data[i]);  
    }

    stbi_image_free(data);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fatal("Usage: ./nano-nn <imagefile>");
    }
    
    char *image_path = argv[1];
    double image[IMAGE_WIDTH * IMAGE_HEIGHT];

    loadImage(image_path, image);

    Dataset *train_data = malloc(sizeof(Dataset));
    Dataset *val_data = malloc(sizeof(Dataset));

    if (!train_data || !val_data) {
         fatal("Malloc failed");
    }

    loadDataset(train_data, TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, TRAIN_COUNT);
    loadDataset(val_data, TEST_IMAGES_PATH, TEST_LABELS_PATH, TEST_COUNT);
    
    NeuralNetwork *n = malloc(sizeof(NeuralNetwork));

    srand(time(NULL)); // random seed
    initializeNeuralNetwork(n);

    trainModel(n, train_data, val_data, EPOCHS);

    // predict image
    double h1[HIDDEN_SIZE_1];
    double h2[HIDDEN_SIZE_2];

    // convert the output into probabilities
    double probs[OUTPUT_SIZE];
    forwardPropagation(image, n, probs, h1, h2, 0);

    // show results
    printf("\n");
    int predicted_digit = 0;
    double max_prob = probs[0];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (probs[i] > max_prob) {
            predicted_digit = i;
            max_prob = probs[i];
        }
        printf("%d --> %f%%\n", i, probs[i]);
    }

    printf("\nThe digit is probably %d with %.2f%% confidence.\n", predicted_digit, max_prob * 100);

    freeDataset(train_data);
    freeDataset(val_data);
    freeNeuralNetwork(n);

    return 0;
}
