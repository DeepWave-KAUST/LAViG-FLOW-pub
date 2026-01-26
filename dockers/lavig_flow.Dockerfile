# syntax=docker/dockerfile:1.6

# Build from repo root:
#   docker build -f dockers/lavig_flow.Dockerfile -t lavig-flow .

FROM mambaorg/micromamba:1.5.8

# These args are provided by the base image but we redeclare them so they can be
# referenced inside COPY instructions when the build starts.
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000

ARG ENV_NAME=lavig-flow
ARG ENV_FILE=dockers/environment.yaml

# micromamba stores environments under /opt/conda by default
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH

WORKDIR /workspace

# Keep environment creation cache-friendly by copying just the spec first.
COPY --chown=${MAMBA_USER}:${MAMBA_USER} ${ENV_FILE} /tmp/environment.yaml

RUN micromamba env create -y -n ${ENV_NAME} -f /tmp/environment.yaml && \
    micromamba clean --all --yes

# Copy the project after the environment is ready so that rebuilding the env
# only happens when the spec changes.
COPY --chown=${MAMBA_USER}:${MAMBA_USER} . .

# Run every container command inside the conda environment.
ENTRYPOINT ["micromamba", "run", "-n", "lavig-flow"]
CMD ["python"]
