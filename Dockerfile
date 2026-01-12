# Use PyTorch 2.9.1 official image with CUDA 12.8 support for RTX 50 series (Blackwell)
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
# Allow PyTorch to work with newer GPUs
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ENV PYTORCH_JIT=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install JupyterLab and additional packages
RUN pip install --no-cache-dir \
    jupyterlab \
    ipywidgets \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    pillow \
    scikit-learn \
    opencv-python-headless \
    tqdm

# Set working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Default command to launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
