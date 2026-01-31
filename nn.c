#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

typedef struct {
    size_t count; // The number of inner layers
    Mat *ws; // arbitrary collection of weight matrices
    Mat *bs; // arbitrary collection of bias matrices
    Mat *as; // collection of activations. it is count+1 because we 
    // adding an extra layer for the inputs.
} NN;

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

// size_t arch[] = {2, 2, 1} // 2-inputs, 2-neurons hidden layer, and 1-output
// NN nn = nn_alloc(arch, ARRAY_LEN(arch))

NN nn_alloc(size_t *arch, size_t arch_count) {
    NN m;
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
    nn.bs = NN_ALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_ALLOC(sizeof(*nn.as)*(nn.count + 1));
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
    }
    return m;
}

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

// Xor g ( g means gradient)
void finite_diff(Xor m, Xor g, float eps, Mat ti, Mat to) {
    float saved;

    float c = cost(m, ti, to);

    // altering the weights matrix of the first layer
    for (size_t i=0; i<m.w1.rows; i++) {
        for (size_t j=0; j<m.w1.cols; j++) {
            saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = ( cost(m, ti, to) - c ) / eps;
            MAT_AT(m.w1, i, j) = saved;
        }
    }
    // altering bias matrix of first layer.
    for (size_t i=0; i<m.b1.rows; i++) {
        for (size_t j=0; j<m.b1.cols; j++) {
            saved = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = ( cost(m, ti, to) - c ) / eps;
            MAT_AT(m.b1, i, j) = saved;
        }
    }

    // altering the weights matrix of the second layer
    for (size_t i=0; i<m.w2.rows; i++) {
        for (size_t j=0; j<m.w2.cols; j++) {
            saved = MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = ( cost(m, ti, to) - c ) / eps;
            MAT_AT(m.w2, i, j) = saved;
        }
    }
    // altering bias matrix of second layer.
    for (size_t i=0; i<m.b2.rows; i++) {
        for (size_t j=0; j<m.b2.cols; j++) {
            saved = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = ( cost(m, ti, to) - c ) / eps;
            MAT_AT(m.b2, i, j) = saved;
        }
    }
}

void learn(Xor m, Xor g, float rate) {
    // updating first layer weights with gradient Xor.
    for (size_t i=0; i<m.w1.rows; i++) {
        for (size_t j=0; j<m.w1.cols; j++) {
            MAT_AT(m.w1, i, j) -= MAT_AT(g.w1, i, j)*rate;
        }
    }
    // updating first layer bias with gradient Xor.
    for (size_t i=0; i<m.b1.rows; i++) {
        for (size_t j=0; j<m.b1.cols; j++) {
            MAT_AT(m.b1, i, j) -= MAT_AT(g.b1, i, j)*rate;
        }
    }
    // updating second layer weights with gradient Xor.
    for (size_t i=0; i<m.w2.rows; i++) {
        for (size_t j=0; j<m.w2.cols; j++) {
            MAT_AT(m.w2, i, j) -= MAT_AT(g.w2, i, j)*rate;
        }
    }
    // updating second layer bias with gradient Xor.
    for (size_t i=0; i<m.b2.rows; i++) {
        for (size_t j=0; j<m.b2.cols; j++) {
            MAT_AT(m.b2, i, j) -= MAT_AT(g.b2, i, j)*rate;
        }
    }
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

    Xor m = xor_alloc();
    Xor g = xor_alloc();

    // we don't need to randomize the activation layers,
    // as they are results of computations.
    mat_rand(m.w1, 0, 1);
    mat_rand(m.b1, 0, 1);
    mat_rand(m.w2, 0, 1);
    mat_rand(m.b2, 0, 1);

    float eps = 1e-1;
    float rate = 1e-1;

    printf("cost = %f\n", cost(m, ti, to));
    for (size_t i=0; i<10*1000; i++) {
        finite_diff(m, g, eps, ti, to);
        learn(m, g, rate);
        printf("%zu: cost = %f\n", i, cost(m, ti, to));
    }

    #if 1
    for (size_t i=0; i<2; i++) {
        for (size_t j=0; j<2; j++) {
            MAT_AT(m.a0, 0, 0) = i;
            MAT_AT(m.a0, 0, 1) = j;
            forward_xor(m);
            float y = *m.a2.es;

            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }
    #endif

    return 0;
}
