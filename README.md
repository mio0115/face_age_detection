<h1 align="center">
Face Age Detection
</h1>

The goal of this project is to detect human face and estimate his/her age from images. We implemented the model "Detection Split Transformer (DESTR)" based on the paper ["Object Detection with Split Transformer"](https://openaccess.thecvf.com/content/CVPR2022/papers/He_DESTR_Object_Detection_With_Split_Transformer_CVPR_2022_paper.pdf) by He, Liqiang, and Sinisa Todorovic. published on CVPR 2022. Then, train on the dataset IMDB-WIKI which has over 300k labeled images. The model is trained with single Nvidia A100 on Lambda Cloud and is containerized with Docker for easy deployment and consistency.

## Table of Contents

- [Setup](#Setup)
- [Train](#Train)
- [Running](#Running-the-Model)
- [License](#license)
- [Contact](#contact)

## Setup
To set up the project, we install Docker on machines. Either build the project from [pulling the Docker image](#Pulling-the-Docker-Image) or [building from source](#Building-from-Sourse). Follow the steps below to get started:
#### Pulling the Docker Image

Pull the pre-built Docker image from the repository.

1.Pull the Docker image:
```bash
# pull the docker image from repository
docker pull dockerhubusername/face_age_detection:latest
```

2.Run the Docker container:
```bash
# Run the container
docker run -it --rm --name face_age_detection_container -v /path/to/local/dataset:/app/dataset yourdockerhubusername/face_age_detection:latest
```

#### Building from Source

1. Clone the repository:

```bash
# clone project
git clone https://github.com/mio0115/face_age_detection.git
cd face_age_detection
```

2.Build the Docker image and run the docker container
```bash
bash scripts/build_and_run.sh
```

## Train

To train the model, ensure the dataset properly set up inside the Docker container. Execute the following command to start the training process:

```bash
bash build_and_run.sh
```

If there are some preferred hyperparameters, change it in scripts/train.sh

## Running the Model

After training, run the model to make predictions on new images. Use the following command to run the inferece script:

```bash
docker exec -it face_age_detection_container python infer.py --image /app/dataset/image/jpg
```