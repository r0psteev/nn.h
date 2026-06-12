// Building a full adder
#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 3 

int main(void) {
    // There (1<<BITS) numbers with 2-bits
    // The total number of possible combinations (additions) between these numbers is
    // (1<<BITS) * (1<<BITS); this is also the number of rows in the training set.
    size_t rows = (1<<BITS) * (1<<BITS);
    Mat ti = mat_alloc(rows, 2*BITS); // a row has 2 numbers, BITS + BITS
    Mat to = mat_alloc(rows, BITS+1); // an additional bit for the overflow bit.
    for (size_t i=0; i<rows; i++) {
        size_t n = i;
        size_t x = 0;
        size_t y = 0;
        size_t z = 0;
        size_t overflow = 0;
        // use `int j`, if you use size_t it will cross the negative line
        // fall into unknown territory
        for (int j=((2*BITS) - 1); j>=0; j--) {
            MAT_AT(ti, i, j) = n%2;
            n /= 2;

            if (j >= BITS) {
                // Isolate first number
                x += (int)MAT_AT(ti, i, j) << (((2*BITS) - 1) - j);
            }else if (j >= 0) {
                // Isolate second number
                y += (int)MAT_AT(ti, i, j) << (((BITS) - 1) - j);
            }

        }
        z = x + y;
        overflow = z >= (1 << BITS); 

        // Put sum in matrix in binary form
        for (int j=BITS-1; j>=0; j--) {
            if (overflow) {
                MAT_AT(to, i, j) = 0;
            }else {
                MAT_AT(to, i, j) = z%2;
                z /= 2;
            }
        }
        MAT_AT(to, i, BITS) = overflow;
    }
    //MAT_PRINT(ti);
    //MAT_PRINT(to);

    // We needed a hidden layer in this network to be able to learn without getting stuck.
    // The number of neurons in the hidden layer (as a rule of thumb) is the next power of 2
    // greater than the number of inputs.
    size_t arch[] = {2*BITS, 8, BITS+1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);
    NN_PRINT(nn);

    float rate = 1;
    printf("cost = %f\n", nn_cost(nn, ti, to));
    // 5000 iterations is not enough for correctness when adder numbers are represented with 3-BITS.
    for (size_t i=0; i<10*1000; i++) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        printf("cost = %f\n", nn_cost(nn, ti, to));
    }

    // count the number of failures and display output
    size_t fails = 0;
    for (size_t i=0; i<rows; i++) {
        size_t n = i;
        size_t x = 0;
        size_t y = 0;
        size_t z = 0;

        for (int j=((2*BITS) - 1); j>=0; j--) {
            MAT_AT(NN_INPUT(nn), 0, j) = n%2;
            n /= 2;

            if (j >= BITS) {
                // Isolate first number
                x += (int)MAT_AT(ti, i, j) << (((2*BITS) - 1) - j);
            }else if (j >= 0) {
                // Isolate second number
                y += (int)MAT_AT(ti, i, j) << (((BITS) - 1) - j);
            }
        }

        z = x + y;
        nn_forward(nn);
        // If our model got  an overflow
        if (MAT_AT(NN_OUTPUT(nn), 0, BITS) >= 0.5f) {
            // if z is less than the max number you can write in BITS bits + 1
            // then, there wasn't truly an overflow.
            if (z < (1 << BITS)) {
                // (OVERFLOW <> %zu): means expected OVERFLOW, got %zu
                printf("%zu + %zu = (OVERFLOW<>%zu)\n", x, y, z);
                fails += 1;
            }
        }else {
            // Check if activation (predicted) sum is difference from target sum z.
            size_t a = 0;
            for (int j=(BITS-1); j>=0; j--) {
                size_t bit = MAT_AT(NN_OUTPUT(nn), 0, j) > 0.5f;
                // convert binary row output to decimal.
                a |= (bit << ((BITS - 1) - j));
            }
            if (z != a) {
                printf("%zu + %zu = (%zu<>%zu)\n", x, y, z, a);
                fails += 1;
            }
        }

        for (size_t k=0; k<NN_INPUT(nn).cols; k++) {
            //int bits = MAT_AT(NN_INPUT(nn), 0, k) >= 0.5;
            int bits = MAT_AT(NN_INPUT(nn), 0, k);
            printf("%2d", bits);
        }
        printf(" = ");
        for (size_t k=0; k<NN_OUTPUT(nn).cols; k++) {
            int bits = MAT_AT(NN_OUTPUT(nn), 0, k) >= 0.5f;
            printf("%2d", bits);
        }
        printf("\n");
    }

    if (fails == 0) printf("OK\n");
}