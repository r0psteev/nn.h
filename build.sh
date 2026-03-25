#!/bin/sh

set -xe
clang -g -Wall -Wextra -o xor xor.c -lm
# -g adds debug symbols for gdb so that you can use `list` command.
