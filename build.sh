#!/bin/shell

docker buildx build --platform linux/amd64 --tag $DOCKER_REGISTRY/alfs/char-recog --push .
# docker buildx build --platform linux/arm64 --tag $DOCKER_REGISTRY/alfs/char-recog --push . -f Dockerfile.l4t
