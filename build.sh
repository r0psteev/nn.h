#!/bin/sh

set -xe

CFLAGS="-g -Wall -Wextra `pkg-config --cflags raylib`"
LIBS="-lm -lX11 `pkg-config --libs raylib`"

# clang -g -Wall -Wextra -o xor xor.c -lm
# -g adds debug symbols for gdb so that you can use `list` command.
clang $CFLAGS -o adder adder.c $LIBS
#clang -Wall -Wextra -o dump_nn dump_nn.c -lm