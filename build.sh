#!/bin/sh

set -xe
#clang -g -Wall -Wextra -o xor xor.c -lm
# -g adds debug symbols for gdb so that you can use `list` command.
#clang -g -Wall -Wextra -o adder adder.c -lm
clang -Wall -Wextra -o dump_nn dump_nn.c -lm
