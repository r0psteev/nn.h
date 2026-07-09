#define NN_IMPLEMENTATION
#include "nn.h"
#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMG_WIDTH 800
#define IMG_HEIGHT 600

uint32_t img_pixels[IMG_WIDTH*IMG_HEIGHT];

int main(void) {
    size_t arch[] = {4, 5, 2, 1}; // XOR
    size_t arch_count = ARRAY_LEN(arch);
    NN nn = nn_alloc(arch, arch_count);
    nn_rand(nn, 0, 1);
    NN_PRINT(nn);

    uint32_t neuron_color = 0xFF0000FF;
    uint32_t connection_color = 0xFF00FF00;
    uint32_t background_color = 0xFF181818;
    Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);
    olivec_fill(img, background_color);

    // Draw all layers
    int neuron_radius = 25;
    int layer_border_vpad = 50; // account for border from top and bottom
    int layer_border_hpad = 50; // account for border left and right
    int nn_width = img.width - 2*layer_border_hpad;
    int nn_height = img.height - 2*layer_border_vpad;
    int nn_x = img.width/2 - nn_width/2 ;
    int nn_y = img.height/2 - nn_height/2;
    int layer_hpad = nn_width / arch_count; // used to equally space layers horizontally.
    for (size_t l=0; l<arch_count; l++) {
        int layer_vpad1 = nn_height/arch[l]; // used to equally space neurons within a layer vertically.
        for (size_t i=0; i < arch[l]; ++i) {
            int cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            int cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
            olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);

            // As long as we are not the last layer, we iterate through the next layer
            // and find the coordinates of its neurons, to connect with them
            if (l+1 < arch_count) {
                int layer_vpad2 = nn_height / arch[l+1];
                for (size_t j=0; j < arch[l+1]; ++j) {
                    int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
                    int cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    olivec_line(img, cx1, cy1, cx2, cy2, connection_color);
                }
            }
        }
    }


    // Draw a frame around the canvas
    uint32_t frame_thicc = 10;
    uint32_t frame_color = 0xFFAAAAAA;
    // we draw the frame up to img.width-1 because img.width is the number of pixels,
    // but pixel index goes from 0 to img.width-1 (same logic applies to img.height)
    olivec_frame(img, 0, 0, img.width-1, img.height-1, frame_thicc, frame_color);


    const char *img_file_path = "nn.png";
    // comp is the amount of components
    // 4 for RGBA
    //stbi_write_png(img_file_path, IMG_WIDTH, IMG_HEIGHT, 4, img_pixels, IMG_WIDTH);
    if(!stbi_write_png(img_file_path, img.width, img.height, 4, img_pixels, img.stride * sizeof(uint32_t))) {
        printf("ERROR: could not save file %s\n", img_file_path);
        return 1;
    }

    printf("Saved NN to %s\n", img_file_path);

    return 0;
}