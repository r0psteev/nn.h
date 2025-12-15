/*
 By default, this file will behave as a header file.
 When you will define `NN_IMPLEMENTATION`, within your `main.c`
 it will behave with implementation also.

 This let's you control where implementations go into your program.
 This is how stb-style headers are structured.
*/
#ifndef NN_H_
#define NN_H_
// header file behavior goes here

#include <stddef.h> // for size_t
#include <assert.h>

// with stb-style we can customize our definition of malloc function
// so as to decouple it from libc.
// by default we could make work using libc's malloc
// and then if we want to use a custom implementation for example in an
// embedded system, we can simply use #define NN_MALLOC [our own malloc]
// at the top of the main.c file
#ifndef NN_MALLOC
#include <stdlib.h> // for size_t
#define NN_MALLOC malloc
#endif

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

typedef struct {
    // fields rows, and cols help us workout the structure of matrix
    // represented by the underlying contiguous array.
    size_t rows;
    size_t cols;
    float *es; // start of contiguous array.
} Mat;

Mat mat_alloc(size_t rows, size_t cols);
// Author wants to concentrate all memory allocations in a single
// place `mat_alloc`.
// such that we we'll have to preallocate matrices when we expect that,
// and then pass them to the other functions which will manipulate them.
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION
// implementations go here

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    // sizeof(*m.es) means size of the first element of m.es
    // which is float in this  case.
    // but in case we change es to double type, it will automatically
    // become double here too.
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    assert(m.es != NULL);
    return m;
}

void mat_dot(Mat dst, Mat a, Mat b) {
    // (void) variable; is a cleant, standard C idiom to suppress
    // compiler warning about unused variables
    // used for Early development scaffolding / stub.
    (void) dst;
    (void) a;
    (void) b;
}
void mat_sum(Mat dst, Mat a) {
    (void) dst;
    (void) a;
}
#endif // NN_IMPLEMENTATION