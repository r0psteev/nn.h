#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"


int main(void) {
    srand(time(NULL));
    // Porting the XOR neuron

    // First layer, check out your obsidian notes
    // to see that the weight matrix in first layer of xor neuron is 2x2
    Mat w1 = mat_alloc(2, 2);
    Mat b1 = mat_alloc(1, 2); // bias matrix is 1x2;

    // Second layer
    Mat w2 = mat_alloc(2, 1);
    Mat b2 = mat_alloc(1, 1);

    mat_rand(w1, 5, 10);
    mat_rand(b1, 5, 10);
    mat_rand(w2, 5, 10);
    mat_rand(b2, 5, 10);

    MAT_PRINT(w1);
    MAT_PRINT(b1);
    MAT_PRINT(w2);
    MAT_PRINT(b2);


    return 0;
}