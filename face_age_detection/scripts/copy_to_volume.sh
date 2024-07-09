#!/bin/bash

# Variables
LOCAL_DATA_PATH="/home/daniel/imdb_chunks"
VOLUME_NAME="datasets/tfrecords"

docker run --rm -v $LOCAL_DATA_PATH:/local_data -v $VOLUME_NAME:/data busybox sh -c "cp -r /local_data/* /data/"

echo "Dataset has been copied into the Docker volume: $VOLUME_NAME"