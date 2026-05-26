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
    for (size_t i=0; i<rows; i++) {
        if (i >= 4 && i < 8) {
            MAT_AT(ti, i, 1) = 1;
        }else if (i >= 8 && i < 12) {
            MAT_AT(ti, i, 0) = 1;
        }else if (i >= 12 && i < 16) {
            MAT_AT(ti, i, 0) = 1;
            MAT_AT(ti, i, 1) = 1;
        }

        if (i % 4 == 0) {
            MAT_AT(ti, i, 2) = 0;
            MAT_AT(ti, i, 3) = 0;
        }else if (i % 4 == 1) {
            MAT_AT(ti, i, 2) = 0;
            MAT_AT(ti, i, 3) = 1;
        }else if (i % 4 == 2) {
            MAT_AT(ti, i, 2) = 1;
            MAT_AT(ti, i, 3) = 0;
        }else if (i % 4 == 3) {
            MAT_AT(ti, i, 2) = 1;
            MAT_AT(ti, i, 3) = 1;
        }
    }
    MAT_PRINT(ti);
}