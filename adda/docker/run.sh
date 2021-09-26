#!/bin/sh
. docker/env.sh
docker stop $CONTAINER_NAME
docker run \
  -dit \
  --gpus all \
  -v $PWD:/workspace \
  -p $JUPYTER_PORT:$JUPYTER_PORT \
  --name $CONTAINER_NAME \
  --rm \
  --shm-size $SHM_SIZE \
  $IMAGE_NAME
