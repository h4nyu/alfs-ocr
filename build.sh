#!/bin/shell

docker buildx build --platform linux/amd64 --tag $DOCKER_REGISTRY/alfs/char-recog --push .
