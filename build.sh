#!/bin/sh

set -xe
clang -g -Wall -Wextra -o nn nn.c -lm
# -g adds debug symbols for gdb so that you can use `list` command.