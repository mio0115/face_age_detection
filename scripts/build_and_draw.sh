#!/bin/sh

# VARIABLES
IMAGE_NAME="projects/destr"
IMAGE_TAG="draw"
CONTAINER_NAME="destr_container_draw"
HOST_PORT=5005
CONTAINER_PORT=8888
DATA_VOLUME_NAME="datasets"
MODEL_VOLUME_NAME="models"
DOCKERFILE="Dockerfile_draw"

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
docker run -it --rm --name $CONTAINER_NAME -p "$HOST_PORT:$CONTAINER_PORT" --gpus all --cpus="6" --memory="10g" -v $DATA_VOLUME_NAME:/workspace/data -v $MODEL_VOLUME_NAME:/workspace/models "$IMAGE_NAME:$IMAGE_TAG"

#if [ -n "$SSH_USER" ] && [ -n "$SSH_SERVER" ]; then
#    echo "Setting up SSH tunneling..."
#    ssh -L $HOST_PORT:localhost:$CONTAINER_PORT $SSH_USER@$SSH_SERVER
#fi

echo "Done!"