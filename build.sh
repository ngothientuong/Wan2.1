#!/bin/bash

# filepath: /home/tngo/ngo/projects/models/Wan2.1/build.sh
# Read the version from the version.txt file
VERSION=$(cat VERSION)

source ${HOME}/ngo/projects/personal/profiles/.env_tngo-ai-svcs
source ${HOME}/.bashrc
azurelogin
# Build the Docker image with the specified version
az acr build --timeout 14400 --registry tngodemo1cr \
    --image ttv/wan21/wan2.1:$VERSION \
    --file Dockerfile .
# Print a message indicating the build is complete
echo "Docker image built with version $VERSION"