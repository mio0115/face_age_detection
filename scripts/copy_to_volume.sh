#!/bin/bash

# Variables
LOCAL_DATA_PATH="/home/ubuntu/face_age_detection/imdb_chunks"
VOLUME_NAME="datasets"

docker run --rm -v $LOCAL_DATA_PATH:/local_data -v $VOLUME_NAME:/data busybox sh -c "cp -r /local_data/* /data/"

if [ $? -ne 0]; then
	echo "Copy Failed"
	exit 1
fi

echo "Dataset has been copied into the Docker volume: $VOLUME_NAME"
