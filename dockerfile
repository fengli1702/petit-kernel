FROM rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0


ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm

WORKDIR /workspace


RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git python3-venv pkg-config \
  && rm -rf /var/lib/apt/lists/*


RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install -U pip wheel setuptools

CMD ["/bin/bash"]
