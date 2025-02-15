#include "read_MNIST.h"


unsigned char* read_ubyte_image(const char* filename, int* num_images, int* num_rows, int* num_cols) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(1);
    }

    // Read the magic number
    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = __builtin_bswap32(magic_number); // Convert to little-endian
    printf("Magic number: %d\n", magic_number); // Debug print

    // Read the number of images
    fread(num_images, sizeof(int), 1, file);
    *num_images = __builtin_bswap32(*num_images);
    printf("Number of images: %d\n", *num_images); // Debug print

    // Read the number of rows
    fread(num_rows, sizeof(int), 1, file);
    *num_rows = __builtin_bswap32(*num_rows);
    printf("Number of rows: %d\n", *num_rows); // Debug print

    // Read the number of columns
    fread(num_cols, sizeof(int), 1, file);
    *num_cols = __builtin_bswap32(*num_cols);
    printf("Number of columns: %d\n", *num_cols); // Debug print

    // Allocate memory for the images
    int image_size = *num_rows * *num_cols;
    unsigned char* images = (unsigned char*)malloc((*num_images) * image_size * sizeof(unsigned char));
    if (images == NULL) {
        fprintf(stderr, "Cannot allocate memory for images\n");
        exit(1);
    }

    // Read the image data
    fread(images, sizeof(unsigned char), (*num_images) * image_size, file);

    fclose(file);
    return images;
}

unsigned char* read_labels(const char* filename, int* num_labels) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(1);
    }

    // Read the magic number
    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = __builtin_bswap32(magic_number); // Convert to little-endian
    printf("Magic number: %d\n", magic_number); // Debug print

    // Read the number of labels
    fread(num_labels, sizeof(int), 1, file);
    *num_labels = __builtin_bswap32(*num_labels);
    printf("Number of labels: %d\n", *num_labels); // Debug print

    // Allocate memory for the labels
    unsigned char* labels = (unsigned char*)malloc((*num_labels) * sizeof(unsigned char));
    if (labels == NULL) {
        fprintf(stderr, "Cannot allocate memory for labels\n");
        exit(1);
    }

    // Read the label data
    fread(labels, sizeof(unsigned char), *num_labels, file);

    fclose(file);
    return labels;
}

// function to flattern the image
void flattern_image(unsigned char* images, int num_images, int num_rows, int num_cols, unsigned char* flattern_images) {
    int total_size = num_images * num_rows * num_cols;
    for (int i = 0; i < total_size; i++) {
        flattern_images[i] = images[i];
    }

    
}

void test_read_MNIST() {
    const char* train_images_filename = TRAINING_IMAGES;
    const char* train_labels_filename = TRAINING_LABELS;
    int num_images, num_rows, num_cols;
    unsigned char* images = read_ubyte_image(train_images_filename, &num_images, &num_rows, &num_cols);

    printf("Number of images: %d\n", num_images);
    printf("Number of rows: %d\n", num_rows);
    printf("Number of columns: %d\n", num_cols);

    // Print the first image
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("%3d ", images[i * num_cols + j]);
        }
        printf("\n");
    }

    // Allocate memory for the flattened images
    unsigned char* flattern_images = (unsigned char*)malloc(num_images * num_rows * num_cols * sizeof(unsigned char));
    if (flattern_images == NULL) {
        fprintf(stderr, "Cannot allocate memory for flattened images\n");
        exit(1);
    }

    // Print the first flatterned image
    flattern_image(images, num_images, num_rows, num_cols, flattern_images);
    for (int i = 0; i < num_rows * num_cols; i++) {
        printf("%3d ", flattern_images[i]);
    }

    int num_labels;
    unsigned char* labels = read_labels(train_labels_filename, &num_labels);

    printf("Number of labels: %d\n", num_labels);

    // Print the first 10 labels
    for (int i = 0; i < 10; i++) {
        printf("Label %d: %d\n", i, labels[i]);
    }

    free(images);
    free(labels);
    free(flattern_images);
}