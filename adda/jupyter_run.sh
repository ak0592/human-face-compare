#!/bin/sh
. $WORK_PATH/docker/env.sh
jupyter lab \
    --port $JUPYTER_PORT \
    --ip=0.0.0.0 \
    --allow-root
