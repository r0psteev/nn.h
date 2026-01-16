#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

typedef struct {
    Mat a0; // inputs
    Mat w1, b1, a1; // 1st layer
    Mat w2, b2, a2; // 2nd layer
} Xor;

float forward_xor(Xor m) {
    mat_dot(m.a1, m.a0, m.w1); // multiplication
    mat_sum(m.a1, m.b1); // bias
    //MAT_PRINT(m.a1);
    mat_sig(m.a1); // activation

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    //MAT_PRINT(m.a2);
    mat_sig(m.a2);

    return *m.a2.es;
}

// ti: train-input
// to: train-output
float cost(Xor m, Mat ti, Mat to) {
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == m.a2.cols);
    size_t n = ti.rows;

    float c = 0;
    for (size_t i=0; i<n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(m.a0, x);
        forward_xor(m);

        size_t q = to.cols;
        // I don't understand the following.
        // if the output has many rows how do we do ?
        // does the output even have many columns ?
        // any ways we are comparing distance of values between the final layer a2,
        // and the expected train output y.
        for (size_t j=0; j<q; j++){
            float d = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
            c += (d*d);
        }
    }
    return c/n;
}


// training data
float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

int main(void) {
    srand(time(NULL));
    // Porting the XOR neuron

    size_t stride = 3;
    size_t n = (sizeof(td) / sizeof(td[0])) / stride; // td / td[0] gives 12, because there are 3 elements on each row
    printf("n = %zu\n", n);

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td,
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };

    MAT_PRINT(ti);
    MAT_PRINT(to);

    return 0;

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
    mat_rand(m.w1, 0, 1);
    mat_rand(m.b1, 0, 1);
    mat_rand(m.w2, 0, 1);
    mat_rand(m.b2, 0, 1);

    for (size_t i=0; i<2; i++) {
        for (size_t j=0; j<2; j++) {
            MAT_AT(m.a0, 0, 0) = i;
            MAT_AT(m.a0, 0, 1) = j;
            forward_xor(m);
            float y = *m.a2.es;

            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }

    return 0;
}
