FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install JupyterLab and additional packages
RUN pip install --no-cache-dir \
    jupyterlab \
    tensorflow-datasets \
    keras

# Set working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Default command to launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
