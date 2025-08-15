# syntax = docker.io/docker/dockerfile-upstream:1.14.1
ARG REGISTRY=habitatsacradptinternal.azurecr.io
FROM ${REGISTRY}/ubuntu-24.04-python-3:2025-03-19.00-10-20 AS base

ENV PROJECT_NAME="custom_projects"
ENV APP_NAME="rfdiffusion"

ENV PROJECT_PATH="/opt/$PROJECT_NAME"
ENV APP_PATH="$PROJECT_PATH/$APP_NAME"

RUN mkdir -p $PROJECT_PATH/.tmp
RUN mkdir -p $APP_PATH/.tmp

WORKDIR $APP_PATH

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN <<HEREDOC
    set -eux

    apt-get update

    # no cudatoolkit results in no cuda runtime found on setup.py install
    apt-get install -yqq --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        wget

    apt-get clean
    rm -rf /var/lib/apt/lists/*
    rm -rf /tmp/*
HEREDOC

FROM base AS builder

# =========== BEGIN: Local Dependencies ===========

COPY --link $APP_NAME/env/SE3Transformer/setup.py $APP_PATH/env/SE3Transformer/setup.py

# Mock out the package directories needed for the local packages
RUN <<HEREDOC
    set -eux

    mkdir -p $APP_PATH/rfdiffusion
    mkdir -p $APP_PATH/env/SE3Transformer/se3_transformer

    touch $APP_PATH/rfdiffusion/__init__.py
    touch $APP_PATH/env/SE3Transformer/se3_transformer/__init__.py
HEREDOC

# We just set this in the builder so we fail if we're not running on the right node type
ENV CONDA_OVERRIDE_CUDA=12.4

RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/rattler \
    --mount=type=bind,source=${APP_NAME}/pixi.lock,target=${APP_PATH}/pixi.lock \
    --mount=type=bind,source=${APP_NAME}/pyproject.toml,target=${APP_PATH}/pyproject.toml <<HEREDOC
    set -eux

    export POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
    export UV_DYNAMIC_VERSIONING_BYPASS=0.0.0

    pixi install --locked --environment production
HEREDOC

# =========== END: Local Dependencies ===========

FROM base

# Install rclone + az cli
RUN <<HEREDOC
    set -eux

    # TODO - no pipes to bash :(
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash
HEREDOC

# TODO - avoid duplicating this w/o rebuilding too often
COPY --link immunoproject.yaml $PROJECT_PATH/immunoproject.yaml


# Copy in the pixi environment
COPY --from=builder $APP_PATH/.pixi/envs/production $APP_PATH/.pixi/envs/production
ENV PATH=$APP_PATH/.pixi/envs/production/bin:$PATH
ENV CONDA_PREFIX=$APP_PATH/.pixi/envs/production

# =========== BEGIN: Local Package Code ===========

# Copy in the full contents of the app itself (after `poetry install`,
# to not invalidate that slow step when local contents change)
COPY --link $APP_NAME $APP_PATH

ENV PATH=$APP_PATH/bin:$PATH

# =========== END: Local Package Code ===========
