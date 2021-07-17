FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
MAINTAINER maomao
WORKDIR /app
COPY . /app
RUN apt-get update -y && apt-get install nginx -y
RUN apt install curl && apt install python3-pip && apt install wget
RUN pip3 install -r requirements.txt
CMD add-apt-repository ppa:graphics-drivers/ppa
CMD ubuntu-drivers autoinstall
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
            && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
RUN apt-get update && apt-get install -y nvidia-docker2
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
CMD sh cuda_11.0.3_450.51.06_linux.run
RUN apt install libgl1-mesa-glx && apt install libglib2.0-0
EXPOSE 80
