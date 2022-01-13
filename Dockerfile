# Install NVIDIA GPU image
FROM nvcr.io/nvidia/l4t-tensorflow:r32.4.4-tf2.3-py3

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common python3-pip python3-opencv python3-setuptools libedit-dev ffmpeg git cmake mpg321 && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip3 install --upgrade pip
RUN pip3 install soundfile soundcard

# Install pywaggle from specific release
# RUN pip3 install git+https://github.com/waggle-sensor/pywaggle.git@0.51.1

# Import all scripts
COPY . ./
RUN pip3 install --no-cache-dir -r requirements.txt

ARG SAGE_STORE_URL="https://osn.sagecontinuum.org" \
    BUCKET_ID_MODEL="cafb2b6a-8e1d-47c0-841f-3cad27737698"

ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    SAGE_STORE_URL=${SAGE_STORE_URL} \
    BUCKET_ID_MODEL=${BUCKET_ID_MODEL}

RUN sage-cli.py storage files download ${BUCKET_ID_MODEL} \
  BirdNET_6K_GLOBAL_MODEL.tflite --target /BirdNET_6K_GLOBAL_MODEL.tflite
# Add entry point to run the script
ENTRYPOINT [ "python3", "./analyze.py" ]

