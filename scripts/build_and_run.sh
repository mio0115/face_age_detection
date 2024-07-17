#!/bin/sh

# VARIABLES
IMAGE_NAME="projects/destr"
IMAGE_TAG="train"
CONTAINER_NAME="destr_container_train"
HOST_PORT=5005
CONTAINER_PORT=8888
DATA_VOLUME_NAME="datasets"
MODEL_VOLUME_NAME="models"
DOCKERFILE="Dockerfile"

# Build image
echo "Building the Docker image..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" -f "$DOCKERFILE" .
docker image prune --filter "dangling=true"

# Check if the image was built successfully
if [ $? -ne 0 ]; then
    echo "Failed to build the Docker image."
    exit 1
fi

# Check if the container is already running and stop it
if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
    echo "Stopping the existing container..."
    docker stop $CONTAINER_NAME
fi

# Run the docker container
docker run -it --rm --name $CONTAINER_NAME --gpus all --cpus="6" --memory="15g" -v $DATA_VOLUME_NAME:/workspace/data -v $MODEL_VOLUME_NAME:/workspace/models "$IMAGE_NAME:$IMAGE_TAG"

echo "Done!"
