#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"


int main(void) {
    srand(time(NULL));
    Mat a = mat_alloc(3, 4);
    mat_rand(a, 0, 10);
    mat_print(a);
    return 0;
}