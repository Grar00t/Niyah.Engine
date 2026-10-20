# Base44 sandbox toolchain for Niyah.Engine.
#
# Toolchain only: no repository source is copied into the image, so the
# container always compiles and runs the bind-mounted working tree.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
