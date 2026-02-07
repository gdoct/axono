# feature: Docker support

## Overview
This feature adds support for running Axono in a Docker container, allowing users to easily set up and run the application without worrying about Python dependencies or environment configuration. The Docker image will include all necessary dependencies and can be run on any system with Docker installed.

## Dockerfile
A `Dockerfile` is included in the repository that defines the Docker image for Axono. It uses a lightweight Python base image, installs the required dependencies, and sets up the application to run when the container starts.

## Scripts
The following scripts will be available:
- `docker-config.sh`: Configures the Docker environment, including building the image and setting up necessary volumes for configuration and data persistence.
- `docker-build.sh`: Builds the Docker image for Axono.
- `docker-run.sh`: Runs the Axono container, mapping necessary ports and volumes for configuration and data persistence.

# Persistence
Axono needs to retain some files. These are stored in the ~/.axono directory on the host machine, which is mounted as a volume in the Docker container. This allows users to keep their configurations and data even after the container is stopped or removed.

# Remote Access
Users should have remote access to Axono. We use ttyd for this purpose, which provides a web-based terminal interface. Example:

```dockerfile
RUN apt-get update && apt-get install -y ttyd

CMD ["ttyd", "-W", "-p", "7681", "-c", "user:pass", "python", "-m", "axono"]
```