#!/bin/sh

# docker buildx build --platform linux/amd64 --tag $DOCKER_REGISTRY/alfs/char-recog --push . -f Dockerfile
docker buildx build --platform linux/arm64 --tag $DOCKER_REGISTRY/alfs/char-recog  . -f Dockerfile.l4t --push
