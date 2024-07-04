FROM tensorflow/tensorflow:latest-gpu-jupyter AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists

RUN pip install --upgrade pip
RUN pip install numpy pandas matplotlib scikit-learn jupyterlab opencv-python-headless


FROM base

RUN mkdir /workspace
WORKDIR /workspace

COPY . /workspace

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]