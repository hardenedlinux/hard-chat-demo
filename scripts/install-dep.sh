#!/bin/bash

export CMAKE_ARGS="-DLLAMA_CUDA=ON"
export FORCE_CMAKE=1

pip3 install -r "$@"
