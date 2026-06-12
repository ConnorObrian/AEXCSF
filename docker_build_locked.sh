#!/bin/bash
docker build --ulimit nofile=5000:5000 -t cschoenberner/aexcsf -f Dockerfile .