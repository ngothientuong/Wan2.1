#!/bin/bash

# filepath: /home/tngo/ngo/projects/models/Wan2.1/build.sh
# Read the version from the version.txt file
VERSION=$(cat VERSION)

source ${HOME}/ngo/projects/personal/profiles/.env_tngo-ai-svcs
source ${HOME}/.bashrc
azurelogin

# ## Build and push without timedout
# ## Build the Docker image locally
docker build -t ttv/wan21/wan2.1:$VERSION -f Dockerfile .

# Login to Azure Container Registry
az acr login --name tngodemo1cr

# Tag the Docker image for ACR
docker tag ttv/wan21/wan2.1:$VERSION tngodemo1cr.azurecr.io/ttv/wan21/wan2.1:$VERSION

# Push the Docker image to ACR
docker push tngodemo1cr.azurecr.io/ttv/wan21/wan2.1:$VERSION

# ## Build the Docker image with the specified version directly with ACR
# az acr build \
#     --registry tngodemo1cr \
#     --image ttv/wan21/wan2.1:$VERSION \
#     --file Dockerfile \
#     . \
#     --timeout 7200
# Print a message indicating the build is complete
echo "Docker image built with version $VERSION"