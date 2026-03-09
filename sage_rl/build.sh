#!/usr/bin/env bash
set -euo pipefail

CXXFLAGS=(
  -O2
  -march=x86-64
  -mtune=generic
  -fcf-protection=none
  -mno-shstk
)

g++ "${CXXFLAGS[@]}" -pthread src/mem-manager.cc src/flow.cc -o mem-manager
g++ "${CXXFLAGS[@]}" -pthread src/sage.cc src/flow.cc -o sage
g++ "${CXXFLAGS[@]}" src/client.c -o client
mv mem-manager rl_module/
mv client sage rl_module/

mkdir -p log
mkdir -p dataset
