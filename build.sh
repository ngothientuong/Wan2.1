#!/bin/bash

# filepath: /home/tngo/ngo/projects/models/Wan2.1/build.sh
# Read the version from the version.txt file
VERSION=$(cat VERSION)

source ${HOME}/ngo/projects/personal/profiles/.env_tngo-ai-svcs
source ${HOME}/.bashrc
azurelogin
# Build the Docker image with the specified version
# Create a build task with a timeout
az acr task create \
    --registry tngodemo1cr \
    --name wan21buildtask \
    --image ttv/wan21/wan2.1:$VERSION \
    --file Dockerfile \
    --context . \
    --timeout 14400
# Print a message indicating the build is complete
echo "Docker image built with version $VERSION"