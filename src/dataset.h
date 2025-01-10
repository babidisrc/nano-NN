#ifndef __DATASET_H_
#define __DATASET_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arpa/inet.h>

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

#define TRAIN_COUNT 60000
#define TEST_COUNT 10000

#define BATCH_SIZE 128

#define TRAIN_IMAGES_PATH "dataset/train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH "dataset/train-labels.idx1-ubyte"
#define TEST_IMAGES_PATH "dataset/t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH "dataset/t10k-labels.idx1-ubyte"

#define PIXEL_SCALE(x) (((double) (x)) / 255.0)

typedef struct Dataset {
    int32_t magic_number_images;
    int32_t numfiles_images;
    int32_t rows;
    int32_t columns;
    uint8_t *pixels;

    int32_t magic_number_labels;
    int32_t numfiles_labels;
    uint8_t *labels; 
} Dataset;

void fatal(char *msg);
FILE* openFile(const char *filepath, const char *mode);

void loadDatasetImgs (Dataset *data, FILE *fh, char filepath[]);
void loadDatasetLabels (Dataset *data, FILE *fh, char filepath[]);
void loadDataset(Dataset *data, char *images_path, char *labels_path, int imageCount);
void freeDataset(Dataset *data);

int createBatch(Dataset *dataset, Dataset *batch, int size, int number);

void printDatasetInfo(const Dataset *data, const char *dataset_name);
void printDatasetSample(Dataset *data);

#endif