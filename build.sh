#!/bin/sh
ARCH=$(uname -m)
echo $ARCH
case $ARCH in
  x86_64)   docker buildx build --platform linux/amd64 --tag alfs-ocr --push . -f Dockerfile ;;
  aarch64)   docker buildx build --platform linux/arm64 --tag alfs-ocr --push . -f Dockerfile.l4t ;;
  *)        echo unexpected arch $ARCH ;;
esac
