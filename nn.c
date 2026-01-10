#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"


int main(void) {
    srand(time(NULL));
    // Porting the XOR neuron

    Mat x = mat_alloc(1, 2);

    // First layer, check out your obsidian notes
    // to see that the weight matrix in first layer of xor neuron is 2x2
    Mat w1 = mat_alloc(2, 2);
    Mat b1 = mat_alloc(1, 2); // bias matrix is 1x2;
    Mat a1 = mat_alloc(1, 2);

    // Second layer
    Mat w2 = mat_alloc(2, 1);
    Mat b2 = mat_alloc(1, 1);
    Mat a2 = mat_alloc(1, 1);

    mat_rand(w1, 5, 10);
    mat_rand(b1, 5, 10);
    mat_rand(w2, 5, 10);
    mat_rand(b2, 5, 10);

    float x1 = 0;
    float x2 = 1;

    MAT_AT(x, 0, 0) = x1;
    MAT_AT(x, 0, 1) = x2;
    // feed forward
    // sigmoidf(x*w1 + b)
    // we are not yet using backpropagation.
    mat_dot(a1, x, w1);
    mat_sum(a1, b1);
    mat_sig(a1);

    mat_dot(a2, a1, w2);
    mat_sum(a2, b2);
    mat_sig(a2);

    float y = *a2.es;

    MAT_PRINT(w1);
    MAT_PRINT(b1);
    MAT_PRINT(w2);
    MAT_PRINT(b2);


    return 0;
}
