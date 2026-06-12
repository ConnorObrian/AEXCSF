
FROM ubuntu:22.04
ENV PYTHON_VERSION=3.13
# Pass your own UID and name! :)
ARG USER_UID=5014
ARG USER_GID=5001
ARG NAME=cschoenberner

LABEL description="PyTorch 2.x + with Python 3.13.x based on the  Ubuntu 22.04 Image"
LABEL maintainer="Connor Schönberner <cos@informatik.uni-kiel.de>"
LABEL version="1.3"
# Builds on tips from https://github.com/astral-sh/uv-docker-example/blob/main/standalone.Dockerfile
# and from https://github.com/astral-sh/uv/issues/7758#issuecomment-2687525578

# USERS
RUN groupadd --gid ${USER_GID} ins
RUN useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${NAME}

WORKDIR /tmp

# Dependencies to build some of the dependencies :)
# ViZDoom dependencies for now commented
# libomp can be useful if building xcsf <libomp-dev> install if needed in the future!
# Usability tools
# Proposal for liter dependencies but one important of the old ones is missing and leads to errors/warning when running FL
# RUN apt-get update -y && apt-get upgrade -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --fix-missing \
# # VizDoom Depens for from source: build-essential cmake git libboost-all-dev libsdl2-dev libopenal-dev python3-dev python3-pip \
# python3-dev python3-pip swig libsdl2-2.0-0 libsdl2-dev \
# zlib1g-dev libffi-dev libsqlite3-dev liblzma-dev  \
# tar wget curl tmux 

# Dependencies to build some of the dependencies :)
# libnss3-dev libssl-dev are probably not needed any longer
RUN apt-get update -y && apt-get upgrade -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --fix-missing \
autoconf autotools-dev automake build-essential bzip2 ca-certificates cmake curl dpkg git g++ gcc libgdbm-dev  \
make perl pkg-config python3-dev python3-pip tar wget zlib1g-dev libffi-dev libsqlite3-dev liblzma-dev tmux


# Configure the Python and UV directory so it is consistent and accessible
ENV UV_PYTHON_INSTALL_DIR="/usr/share/uv/python"
ENV UV_INSTALL_DIR="/usr/local/bin"

# Install uv for Python and dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Ensure the installed binary is on the `PATH` - superfluous?
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /home/$NAME/aexcsf

# Enable bytecode compilation 
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Only use the managed Python version
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python before the project for caching
RUN uv python install $PYTHON_VERSION
COPY pyproject.toml uv.lock .

# Install the project's dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
--mount=type=bind,source=pyproject.toml,target=pyproject.toml \
uv sync --locked --no-install-project --no-dev

COPY --chown=${NAME}:${USER_GID} . .
# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
RUN --mount=type=cache,target=/root/.cache/uv \
uv sync --locked --no-dev

# Next line is needed so that CUBLAS runs deterministically which is used by NNs seemingly
ENV CUBLAS_WORKSPACE_CONFIG=":4096:8"

# CLEANUP
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Place executables in the environment at the front of the path
ENV PATH="/home/$NAME/aexcsf/.venv/bin:$PATH"

# Use the non-root user to run our application
USER $NAME