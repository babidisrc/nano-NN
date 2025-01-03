#ifndef __PARSER_H_
#define __PARSER_H_

#define TRAIN_RATIO 0.8
#define VAL_RATIO 0.1

#define MAX_SAMPLES 2000
#define MAX_HEADERS 50

typedef struct {
    int size;
    int input_size;
    int* labels;
    double** data;
} Dataset;

Dataset* createDataset(int size, int input_size) ;
void freeDataset(Dataset* dataset);
void splitDataset(Dataset** train, Dataset** val, Dataset** test, int samples, int input_size, double data[][MAX_HEADERS], int all_y_trues[]);

#endif