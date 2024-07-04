#!/bin/sh

# VARIABLES
IMAGE_NAME="projects"
IMAGE_TAG="destr"
CONTAINER_NAME="destr_container"
HOST_PORT=5005
CONTAINER_PORT=8888
DATA_VOLUME_NAME="datasets"
MODEL_VOLUME_NAME="models"
DOCKERFILE="Dockerfile"
BUILD_METADATA_DIR=".build"
HASH_FILE="$BUILD_METADATA_DIR/last_build.hash"

# Function to generate a hash of the project directory
generate_hash() {
    find . -type f -not -path "./$BUILD_METADATA_DIR/*" -exec md5sum {} + | sort -k 2 | md5sum | awk '{ print $1 }'
}

# Check if the Docker image needs to be rebuilt

# Generate the current bash of the project directory
CURRENT_HASH=$(generate_hash)

if [ $? -ne 0 ]; then
    echo "Failed to generate hashes."
    exit 1
fi

# Get the hashes generated before
if [ -f $HASH_FILE ]; then
    LAST_HASH=$(cat $HASH_FILE)
else
    LAST_HASH=""
fi

# Compare the hashes and rebuild the image if they differ
if [ "$CURRENT_HASH" != "$LAST_HASH" ]; then
    echo "Changes detected, building the docker image..."
    docker build -t "$IMAGE_NAME:$IMAGE_TAG" -f "$DOCKERFILE" .

    # Check if the image was built successfully
    if [ $? -ne 0 ]; then
        echo "Failed to build the Docker image."
        exit 1
    fi

    # Update the hash file with the new bash
    echo $CURRENT_HASH > $HASH_FILE
else
    echo "No changes detected, skipping Docker image build."
fi

# Check if the container is already running and stop it
if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
    echo "Stopping the existing container..."
    docker stop $CONTAINER_NAME
fi

# Run the docker container
docker run -it --rm --name $CONTAINER_NAME -p "$HOST_PORT:$CONTAINER_PORT" --gpus all -v $DATA_VOLUME_NAME:/workspace/data -v $MODEL_VOLUME_NAME:/workspace/models "$IMAGE_NAME:$IMAGE_TAG"

#if [ -n "$SSH_USER" ] && [ -n "$SSH_SERVER" ]; then
#    echo "Setting up SSH tunneling..."
#    ssh -L $HOST_PORT:localhost:$CONTAINER_PORT $SSH_USER@$SSH_SERVER
#fi

echo "Done!"
