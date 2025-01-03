#include "core.h"

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

void splitDataset(Dataset** train, Dataset** val, Dataset** test, int samples, int input_size, double data[][MAX_HEADERS], int all_y_trues[]) {
    *train = createDataset((int)(samples * TRAIN_RATIO), input_size);
    *val = createDataset((int)(samples * VAL_RATIO), input_size);
    *test = createDataset((int)(samples - ((*train)->size + (*val)->size)), input_size);

    for (int i = 0; i < samples; i++) {
        all_y_trues[i] = (int)data[i][input_size]; 
    }

    for (int i = 0; i < (*train)->size; i++) {
        for (int j = 0; j < input_size; j++) {
            (*train)->data[i][j] = data[i][j];
        }
        (*train)->labels[i] = all_y_trues[i];
    }

    for (int i = 0; i < (*val)->size; i++) {
        for (int j = 0; j < input_size; j++) {
            (*val)->data[i][j] = data[(*train)->size + i][j];
        }
        (*val)->labels[i] = all_y_trues[(*train)->size + i];
    }

    for (int i = 0; i < (*test)->size; i++) {
        for (int j = 0; j < input_size; j++) {
            (*test)->data[i][j] = data[(*train)->size + (*val)->size + i][j];
        }
        (*test)->labels[i] = all_y_trues[(*train)->size + (*val)->size + i];
    }
}