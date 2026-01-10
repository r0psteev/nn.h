#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

typedef struct {
    Mat a0; // inputs
    Mat w1, b1, a1; // 1st layer
    Mat w2, b2, a2; // 2nd layer
} Xor;

float forward_xor(Xor m, float x1, float x2) {
    MAT_AT(m.a0, 0, 0) = x1;
    MAT_AT(m.a0, 0, 1) = x2;
    
    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);

    return *m.a2.es;
}

int main(void) {
    srand(time(NULL));
    // Porting the XOR neuron

    Xor m;
    m.a0 = mat_alloc(1, 2);
    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2); // bias matrix is 1x2;
    m.a1 = mat_alloc(1, 2);
    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);

    // we don't need to randomize the activation layers,
    // as they are results of computations.
    mat_rand(m.w1, 5, 10);
    mat_rand(m.b1, 5, 10);
    mat_rand(m.w2, 5, 10);
    mat_rand(m.b2, 5, 10);

    return 0;
}
