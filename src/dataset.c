/*
    File format: http://yann.lecun.com/exdb/mnist/
*/

#include "dataset.h"
#include "core.h"

void fatal(char *msg) {
    fprintf(stderr, "Error: %s\n", msg);
    exit(1);
}

FILE* openFile(const char *filepath, const char *mode) {
    FILE *fh = fopen(filepath, mode);
    if (fh == NULL) {
        fprintf(stderr, "Error: File not found: %s\n", filepath);
        exit(1);
    }
    return fh;
}

void loadDatasetImgs (Dataset *data, FILE *fh, char filepath[]) {
    if((fread(&data->magic_number_images, sizeof(int32_t), 1, fh)) != 1) {
        fclose(fh);
        fatal("Couldn't read the file");
    }

    if((fread(&data->numfiles_images, sizeof(int32_t), 1, fh)) != 1) {
        fclose(fh);
        fatal("Couldn't read the file");
    }
    
    fread(&data->rows, sizeof(int32_t), 1, fh);
    fread(&data->columns, sizeof(int32_t), 1, fh);

    // ntohl(): big-endian --> little-endian
    data->magic_number_images = ntohl(data->magic_number_images);

    if (data->magic_number_images != 2051) {
        fatal("Invalid magic number for images file");
    }

    data->numfiles_images = ntohl(data->numfiles_images);
    data->rows = ntohl(data->rows);
    data->columns = ntohl(data->columns);

    fread(data->pixels, sizeof(uint8_t), data->numfiles_images * IMAGE_WIDTH * IMAGE_HEIGHT, fh);

    fclose(fh);
}

void loadDatasetLabels (Dataset *data, FILE *fh, char filepath[]) {
    if((fread(&data->magic_number_labels, sizeof(int32_t), 1, fh)) != 1) {
        fclose(fh);
        fatal("Couldn't read the file");
    }

    if((fread(&data->numfiles_labels, sizeof(int32_t), 1, fh)) != 1) {
        fclose(fh);
        fatal("Couldn't read the file");
    }

    data->magic_number_labels = ntohl(data->magic_number_labels);

    if (data->magic_number_labels != 2049) {
        fatal("Invalid magic number for labels file");
    }

    data->numfiles_labels = ntohl(data->numfiles_labels);

    if (data->numfiles_images != data->numfiles_labels) {
        fatal("Number of images and labels do not match");
    }

    fread(data->labels, sizeof(uint8_t), data->numfiles_labels, fh);

    fclose(fh);
}

void loadDataset(Dataset *data, char *images_path, char *labels_path, int imageCount) {
    // allocate memory for pixels and labels
    data->pixels = malloc(imageCount * IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uint8_t));
    data->labels = malloc(imageCount * sizeof(uint8_t));

    if (!data->pixels || !data->labels) {
        fatal("Failed to allocate memory for dataset");
    }

    // load images
    FILE *fh = openFile(images_path, "rb");
    loadDatasetImgs(data, fh, images_path);

    // load labels
    fh = openFile(labels_path, "rb");
    loadDatasetLabels(data, fh, labels_path);

    // debug: Print dataset size
    printf("Loaded dataset with %d images and %d labels\n", data->numfiles_images, data->numfiles_labels);
}

// free memory
void freeDataset(Dataset *data) {
    free(data->pixels);
    free(data->labels);
    free(data);
}

// create batch to accelerate training process
int createBatch(Dataset *dataset, Dataset *batch, int size, int number) {
    int start_offset = size * number;

    // check if the start offset is valid
    if (start_offset >= dataset->numfiles_images) {
        return 0; // invalid batch number
    }

    // adjust batch size if it exceeds the dataset size
    int remaining_images = dataset->numfiles_images - start_offset;
    int actual_batch_size = (remaining_images < size) ? remaining_images : size;

    // set batch pointers and sizes
    batch->pixels = &dataset->pixels[start_offset * IMAGE_WIDTH * IMAGE_HEIGHT];
    batch->labels = &dataset->labels[start_offset];
    batch->numfiles_images = actual_batch_size;
    batch->numfiles_labels = actual_batch_size;

   // printf("Creating batch: start_offset = %d, actual_batch_size = %d\n", start_offset, actual_batch_size);

    return 1; // success
}

void printDatasetInfo(const Dataset *data, const char *dataset_name) {
    printf("%s DATASET\n", dataset_name);
    printf("Magic number (images): %d\n", data->magic_number_images);
    printf("Number of images: %d\n", data->numfiles_images);
    printf("Image dimensions: %d x %d\n\n", data->rows, data->columns);

    printf("Magic number (labels): %d\n", data->magic_number_labels);
    printf("Number of labels: %d\n", data->numfiles_labels);
    printf("\n\n");

    int i;
    for (i = 0; i < data->numfiles_images; i++) {
        printf("%f ", PIXEL_SCALE(data->pixels[i]));
    }
    printf("\nnumber of images = %d\n", i);
}

void printDatasetSample(Dataset *data) {
    printf("Sample of dataset:\n");
    for (int i = 0; i < 1; i++) {
        printf("Label: %d, Pixels: \n", data->labels[i]);
        for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT; j++) { // Print first 10 pixels
            printf("%f ", PIXEL_SCALE(data->pixels[i * IMAGE_WIDTH * IMAGE_HEIGHT + j]));
        }
        printf("\n");
    }
}