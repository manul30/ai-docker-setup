FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
# Force TensorFlow to JIT-compile CUDA kernels for Blackwell (compute capability 12.0)
ENV TF_CUDA_COMPUTE_CAPABILITIES="12.0"

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install JupyterLab and additional packages
RUN pip install --no-cache-dir \
    jupyterlab \
    tensorflow-datasets \
    keras \
    tf-models-official

# Set working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Default command to launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
