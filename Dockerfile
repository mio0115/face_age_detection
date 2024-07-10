FROM tensorflow/tensorflow:2.16.1-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists
   
RUN mkdir /workspace
WORKDIR /workspace

COPY requirements.txt .
COPY scripts/train.sh .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash", "train.sh"]
