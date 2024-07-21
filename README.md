<h1 align="center">
Face Age Detection
</h1>

The goal of this project is to detect human face and his/her age in an image. We use DESTR model from paper "Object Detection with Split Transformer" and train on the dataset IMDB wiki which has over 300k images. The model is trained on Lambda Cloud with single Nvidia A100.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation
First, install dependencies
```bash
# clone project
git clone https://github.com/mio0115/face_age_detection.git
```
Then, install Docker Engine to build the Docker image
https://docs.docker.com/desktop/install/ .

Then we can build the image and running container based on the image through the script "build_and_run.sh" in scripts/.

For the dataset, we would need to convert the data into tfrecord format and place it into the docker volume named data through the script "copy_to_volume.sh" in scripts/. Since we use ResNet50 as backbone, remember to resize the input images into shape (224, 224, 3).