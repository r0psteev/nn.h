// Building a full adder
#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 2

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
    MAT_PRINT(ti);
    MAT_PRINT(to);
}