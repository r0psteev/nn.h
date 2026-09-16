// Building a full adder
#define NN_IMPLEMENTATION
#include "nn.h"

#define OLIVEC_IMPLEMENTATION
#include "olive.c"

#include "raylib.h"

#define BITS 4 

#define IMG_FACTOR 80
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

uint32_t img_pixels[IMG_WIDTH*IMG_HEIGHT];

void nn_render(Olivec_Canvas img, NN nn)
{

    uint32_t background_color = 0xFF181818;
    uint32_t low_color = 0x00FF00FF; // Red
    uint32_t high_color = 0x0000FF00; // Green

    olivec_fill(img, background_color);

    // Draw all layers
    int neuron_radius = 25;
    int layer_border_vpad = 50; // account for border from top and bottom
    int layer_border_hpad = 50; // account for border left and right
    int nn_width = img.width - 2*layer_border_hpad;
    int nn_height = img.height - 2*layer_border_vpad;
    int nn_x = img.width/2 - nn_width/2 ;
    int nn_y = img.height/2 - nn_height/2;
    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count; // used to equally space layers horizontally.
    for (size_t l=0; l<arch_count; l++) {
        // Each column in an `nn.as[i]` matrix represents a neuron
        // because we have a single activation per neuron in a row-vector
        int layer_vpad1 = nn_height/nn.as[l].cols; // used to equally space neurons within a layer vertically.
        for (size_t i=0; i < nn.as[l].cols; ++i) {
            int cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            int cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;

            // As long as we are not the last layer, we iterate through the next layer
            // and find the coordinates of its neurons, to connect with them
            if (l+1 < arch_count) {
                int layer_vpad2 = nn_height / nn.as[l+1].cols;
                for (size_t j=0; j < nn.as[l+1].cols; ++j) {
                    int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
                    int cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    uint32_t s = floorf(255.f*sigmoidf(MAT_AT(nn.ws[l], j, i)));
                    uint32_t connection_color = 0xFF000000|low_color;
                    olivec_blend_color(&connection_color, (s<<(8*3))|high_color);
                    olivec_line(img, cx1, cy1, cx2, cy2, connection_color);
                }
            }
            if (l > 0) {
                // intensity of bias squished between 0 and 1 and the multiplied to 255
                // this will be used used to define the intensity of green color we want to use.
                uint32_t alpha = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
                uint32_t neuron_color = 0xFF000000|low_color;
                // We blend the 2 colors together to obtain our neuron color
                olivec_blend_color(&neuron_color, (alpha<<(8*3))|high_color);
                // Making sure to draw neurons on top (after) of connections.
                olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);
            } else {
                olivec_circle(img, cx1, cy1, neuron_radius, 0xFFAAAAAA);
            }
        }
    }

}


void nn_render_raylib(NN nn)
{

    // Olivec uint32_t 0xFF181818 is in the order alpha-rgb
    // while raylib is in order rgba.
    Color background_color = {0x18, 0x18, 0x18, 0xFF};
    // We decided to change alpha here because raylib might have a better way of blending colors
    Color low_color = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_color = {0x00, 0xFF, 0x00, 0xFF};

    ClearBackground(background_color);

    // Draw all layers
    int neuron_radius = 25;
    int layer_border_vpad = 50; // account for border from top and bottom
    int layer_border_hpad = 50; // account for border left and right
    int nn_width = IMG_WIDTH - 2*layer_border_hpad;
    int nn_height = IMG_HEIGHT - 2*layer_border_vpad;
    int nn_x = IMG_WIDTH/2 - nn_width/2 ;
    int nn_y = IMG_HEIGHT/2 - nn_height/2;
    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count; // used to equally space layers horizontally.
    for (size_t l=0; l<arch_count; l++) {
        // Each column in an `nn.as[i]` matrix represents a neuron
        // because we have a single activation per neuron in a row-vector
        int layer_vpad1 = nn_height/nn.as[l].cols; // used to equally space neurons within a layer vertically.
        for (size_t i=0; i < nn.as[l].cols; ++i) {
            int cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            int cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;

            // As long as we are not the last layer, we iterate through the next layer
            // and find the coordinates of its neurons, to connect with them
            if (l+1 < arch_count) {
                int layer_vpad2 = nn_height / nn.as[l+1].cols;
                for (size_t j=0; j < nn.as[l+1].cols; ++j) {
                    int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
                    int cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.ws[l], j, i)));
                    DrawLine(cx1, cy1, cx2, cy2, ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }
            if (l > 0) {
                // intensity of bias squished between 0 and 1 and the multiplied to 255
                // this will be used used to define the intensity of green color we want to use.
                high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
                DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
            } else {
                Color input_layer = {0xAA, 0xAA, 0xAA, 0xFF};
                DrawCircle(cx1, cy1, neuron_radius, input_layer);
            }
        }
    }

}

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
    size_t arch[] = {2*BITS, 16, BITS+1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);
    NN_PRINT(nn);

    float rate = 1;

    InitWindow(IMG_WIDTH, IMG_HEIGHT, "adder");
    SetTargetFPS(60);

    int i = 0;
    while (!WindowShouldClose()) {
        if (i < 20*1000) {
            nn_backprop(nn, g, ti, to);
            nn_learn(nn, g, rate);
        }
        i+=1;

        BeginDrawing();
        {
            ClearBackground(RAYWHITE);
            nn_render_raylib(nn);
        }
        EndDrawing();
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