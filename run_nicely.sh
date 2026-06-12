#!/bin/bash

# Data in a normal volume 
# Installation in a normal volume so that you don't need to pull changes and you can use the container as a dev container
# venv in an anonymous volume so that it does not end up in the image see https://github.com/astral-sh/uv-docker-example/blob/main/run.sh

docker run --gpus '"device=2"' --memory=126GB --memory-reservation=32GB --cpus=32 -itd --rm --name aexcsfd2 \
-v ../aexcsf:/home/$USER/aexcsf  -v /home/$USER/aexcsf/.venv cschoenberner/aexcsf bash