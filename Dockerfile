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

# Install JupyterLab and general data-science packages
RUN pip install --no-cache-dir \
    jupyterlab \
    ipywidgets \
    matplotlib \
    seaborn \
    pandas \
    pillow \
    scikit-learn \
    opencv-python-headless \
    tqdm \
    roboflow \
    torch-pruning \
    thop \
    pycocotools

# Install TensorFlow + litert-torch pinned together
# (tensorflow 2.19 needs numpy<2.2 and protobuf==4.25.5;
#  litert-torch is the official PyTorch->TFLite converter)
RUN pip install --no-cache-dir \
    "numpy>=1.26.4,<2.0" \
    "protobuf==4.25.5" \
    "tensorflow==2.19.0" \
    litert-torch

# Install ONNX export stack pinned for mutual compatibility:
#   onnx 1.19.1  <->  onnxruntime 1.23.0  <->  onnx-graphsurgeon 0.5.8
RUN pip install --no-cache-dir \
    onnx==1.19.1 \
    onnxsim==0.4.36 \
    onnxscript \
    onnxruntime==1.23.0 \
    onnx-graphsurgeon==0.5.8 \
    coremltools==9.0

# Set working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Default command to launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
