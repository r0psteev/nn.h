#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"


int main(void) {
    srand(time(NULL));
    Mat a = mat_alloc(2, 2);
    mat_fill(a, 3);

    Mat b = mat_alloc(2, 2);
    mat_fill(b, 1);

    printf("---------------------\n");
    mat_print(a);
    mat_sum(a, b);
    printf("---------------------\n");
    mat_print(a);
    return 0;
}