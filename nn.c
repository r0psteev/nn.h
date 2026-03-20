#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"


void nn_forward(NN nn) {
    for (size_t i=0; i<nn.count; i++) {
        // i+1 does give an overflow on nn.as
        // because the size of nn.as during allocation is nn.count + 1
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
}

// training data
float td_xor[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

float td_or[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

int main(void) {

    srand(time(NULL));
    float *td = td_or;
    
    // Porting the XOR neuron
    
    size_t stride = 3;
    size_t n = 4;
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

    // 2-inputs
    // 2 neurons in first layer
    // 1 neuron in last layer.
    size_t arch[] = {2, 4, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gg = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);

    float eps = 1e-1;
    float rate = 1e-1;

    printf("cost = %f\n", nn_cost(nn, ti, to));
    for (size_t i=0; i<20*1000; i++) {
        nn_finite_diff(nn, gg, eps, ti, to);
        nn_learn(nn, gg, rate);
        printf("%zu: cost = %f\n", i, nn_cost(nn, ti, to));
    }

    for (size_t i=0; i<2; i++) {
        for (size_t j=0; j<2; j++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }
    }

    NN_PRINT(nn);

    return 0;
}
