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
#include <math.h>

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

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float);

typedef struct {
    // fields rows, and cols help us workout the structure of matrix
    // represented by the underlying contiguous array.
    size_t rows;
    size_t cols;
    size_t stride;
    float *es; // start of contiguous array.
} Mat;

// each ith row is (m).cols in length.
// we wrapp m, i, j in brackets to prevent in-place
// replacement of expressions in the macro which could
// alter BODMASS order.
#define  MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
// Author wants to concentrate all memory allocations in a single
// place `mat_alloc`.
// such that we we'll have to preallocate matrices when we expect that,
// and then pass them to the other functions which will manipulate them.
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_print(Mat a, const char* name, size_t padding);
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float num);
void mat_sig(Mat m);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
// the #m stringifies the token m, it converts the variable name m into a
// a string.
#define MAT_PRINT(m) mat_print(m, #m, 0);

// Neural network api
typedef struct {
    size_t count; // The number of inner layers
    Mat *ws; // arbitrary collection of weight matrices
    Mat *bs; // arbitrary collection of bias matrices
    Mat *as; // collection of activations. it is count+1 because we 
    // adding an extra layer for the inputs.
} NN;

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION
// implementations go here

float rand_float(void) {
    // returns a random number from 0 to 1.
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
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
    NN_ASSERT(dst.cols == a.cols);
    for (size_t i=0; i<dst.rows; i++) {
        for (size_t j=0; j<dst.cols; j++) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_print(Mat m, const char* name, size_t padding) {
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; i++) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
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

Mat mat_row(Mat m, size_t row) {
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        // since MAT_AT is an expression that points to an element with es array,
        // using ampersand infront of it gives us a pointer from the element pointed,
        // in this case it is the first element in the row.
        // Thus a pointer to the first element in the row.
        .es = &MAT_AT(m, row, 0),
    };
}

void mat_copy(Mat dst, Mat src) {
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);
    for (size_t i=0; i<dst.rows; i++){
        for (size_t j=0; j<dst.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

// size_t arch[] = {2, 2, 1} // 2-inputs, 2-neurons hidden layer, and 1-output
// NN nn = nn_alloc(arch, ARRAY_LEN(arch))

NN nn_alloc(size_t *arch, size_t arch_count) {
    NN nn;
    nn.count = arch_count - 1; // nn.count is the amount of hidden layers,
    // but arch_count includes the input layer, so we subtract it by 1;
    //
    nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count); //sizeof works at compile time
    // the expression inside it is evaluated as compile time, not runtime.
    // in our case *nn.ws refers to the element nn.ws is pointing to.
    // So are are allocating many matrices with sizeof(*nn.ws), since
    // nn.ws points to an object of type matrix.
    // non-VLA types (Variable Length Array) are arrays whose size we know at
    // compile time.
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    // The input layer is just a 1-row vector, [x1, x2]
    // But whose number of columns is given by `arch[0]`.
    nn.as[0] = mat_alloc(1, arch[0]);
    for (size_t i=1; i < arch_count; i++) {
        // The number of rows for the current weight matrix
        // should be equal to the number of columns of the previous activation matrix
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        // The bias is a vector of 1-row, and its number of columns is the same as arch[1]
        // .ie the number of neurons in the hidden layer.
        nn.bs[i-1] = mat_alloc(1, arch[i]);
        nn.as[i] = mat_alloc(1, arch[i]);
    }
    return nn;
}

void nn_print(NN nn, const char *name) {
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i=0; i<nn.count; i++) {
        snprintf(buf, sizeof(buf), "ws[%zu]", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs[%zu]", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high) {
    for(size_t i=0; i<nn.count; i++) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

#endif // NN_IMPLEMENTATION

