// Building a full adder
#define NN_IMPLEMENTATION
#include "nn.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

#define BITS 4 

#define IMG_FACTOR 80
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)


void nn_render_imgui(NN nn)
{
    // Olivec uint32_t 0xFF181818 is in the order alpha-rgb
    // while raylib is in order rgba.
    ImVec4 background_color = ImVec4(0x18/255.0f, 0x18/255.0f, 0x18/255.0f, 1.0f);
    // We decided to change alpha here because raylib might have a better way of blending colors
    ImVec4 low_color = ImVec4(0xFF/255.0f, 0x00/255.0f, 0xFF/255.0f, 1.0f);
    ImVec4 high_color = ImVec4(0x00/255.0f, 0xFF/255.0f, 0x00/255.0f, 1.0f);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    glClearColor(background_color.x * background_color.w, background_color.y * background_color.w, background_color.z * background_color.w, background_color.w);
    glClear(GL_COLOR_BUFFER_BIT);

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
                    high_color.w = floorf(255.f*sigmoidf(MAT_AT(nn.ws[l], j, i)));
                    ImVec4 blendedVec = ImLerp(low_color, high_color, 1.0f);
                    ImU32 blendedColor = ImGui::ColorConvertFloat4ToU32(blendedVec);
                    draw_list->AddLine(ImVec2(cx1, cy1), ImVec2(cx2, cy2), blendedColor, 1.0f);
                }
            }
            if (l > 0) {
                // intensity of bias squished between 0 and 1 and the multiplied to 255
                // this will be used used to define the intensity of green color we want to use.
                high_color.w = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
                ImVec4 blendedVec = ImLerp(low_color, high_color, 1.0f);
                ImU32 blendedColor = ImGui::ColorConvertFloat4ToU32(blendedVec);
                draw_list->AddCircleFilled(ImVec2(cx1, cy1), neuron_radius, blendedColor);
            } else {
                ImVec4 input_layer_vec = ImVec4(0xAA/255.0f, 0xAA/255.0f, 0xAA/255.0f, 1.0f);
                ImU32 input_layer_color = ImGui::ColorConvertFloat4ToU32(input_layer_vec);
                draw_list->AddCircleFilled(ImVec2(cx1, cy1), neuron_radius, input_layer_color);
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

    // Setup SDL
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD))
    {
        printf("Error: SDL_Init(): %s\n", SDL_GetError());
        return 1;
    }

    const char* glsl_version = nullptr;

    // GL 3.0 + generally GLSL 130
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    float main_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    SDL_WindowFlags window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    SDL_Window* window = SDL_CreateWindow("Dear ImGui SDL3+OpenGL3 example", (int)(1280 * main_scale), (int)(800 * main_scale), window_flags);
    if (window == nullptr)
    {
        printf("Error: SDL_CreateWindow(): %s\n", SDL_GetError());
        return 1;
    }
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (gl_context == nullptr)
    {
        printf("Error: SDL_GL_CreateContext(): %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(window);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;   // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);
    style.FontScaleDpi = main_scale;

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Main loop
    bool done = false;

    int i = 0;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT)
                done = true;
            if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window))
                done = true;
        }

        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED)
        {
            SDL_Delay(10);
            continue;
        }

        if (i < 20*1000) {
            nn_backprop(nn, g, ti, to);
            nn_learn(nn, g, rate);
            printf("cost = %f\n", nn_cost(nn, ti, to));
            i+=1;
        }

        // Start Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("adder");

            //ClearBackground(RAYWHITE);
            //nn_render_raylib(nn);
            nn_render_imgui(nn);

            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        //glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        //glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DestroyContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

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