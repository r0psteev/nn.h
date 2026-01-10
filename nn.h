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
#include <stdio.h>

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
#endif //NN_ASSERT

typedef struct {
    // fields rows, and cols help us workout the structure of matrix
    // represented by the underlying contiguous array.
    size_t rows;
    size_t cols;
    float *es; // start of contiguous array.
} Mat;

// each ith row is (m).cols in length.
// we wrapp m, i, j in brackets to prevent in-place
// replacement of expressions in the macro which could
// alter BODMASS order.
#define  MAT_AT(m, i, j) (m).es[(i)*(m).cols + (j)]

float rand_float(void);
float sigmoidf(float);

Mat mat_alloc(size_t rows, size_t cols);
// Author wants to concentrate all memory allocations in a single
// place `mat_alloc`.
// such that we we'll have to preallocate matrices when we expect that,
// and then pass them to the other functions which will manipulate them.
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_print(Mat a, const char* name);
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float num);
void mat_sig(Mat m);
// the #m stringifies the token m, it converts the variable name m into a
// a string.
#define MAT_PRINT(m) mat_print(m, #m);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION
// implementations go here

float rand_float(void) {
    // returns a random number from 0 to 1.
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + exp(-x));
}

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    // sizeof(*m.es) means size of the first element of m.es
    // which is float in this  case.
    // but in case we change es to double type, it will automatically
    // become double here too.
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_dot(Mat dst, Mat a, Mat b) {
    // 1 x 3 dot 3 x 2 ==> 1 x 2
    // inner sizes must be equal, ie 3 and 3
    NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    // size of dst matrix is outter sizes ie 1 x 2;
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for (size_t i=0; i<dst.rows; i++) {
        for (size_t j=0; j<dst.cols; j++) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k=0; k<n; k++) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}
void mat_sum(Mat dst, Mat a) {
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.rows);
    for (size_t i=0; i<dst.rows; i++) {
        for (size_t j=0; j<dst.cols; j++) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_print(Mat m, const char* name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            printf("    %f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

// low and high is the range for the random numbers that will
// the matrix to be in.
void mat_rand(Mat m, float low, float high) {
    for (size_t i=0; i < m.rows; i++) {
        for (size_t j=0; j<m.cols; j++) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_fill(Mat m, float num) {
    for (size_t i=0; i < m.rows; i++) {
        for (size_t j=0; j<m.cols; j++) {
            MAT_AT(m, i, j) = num;
        }
    }
}

// Goes through all the elements of the matrix and
// applies sigmoid on them.
void mat_sig(Mat m) {
    for (size_t i=0; i < m.rows; i++) {
        for (size_t j=0; j<m.cols; j++) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}
#endif // NN_IMPLEMENTATION
