
FROM nvidia/cuda:9.0-runtime

RUN apt-get update && \
    apt-get -y install build-essential python-dev python3-dev python3-pip python-imaging wget curl

RUN apt-get -y install git

RUN mkdir /opt/program
RUN cd /opt/program

COPY pytorch-CycleGAN-and-pix2pix /tmp/pytorch
RUN mv /tmp/pytorch/* /opt/program/
COPY data-trees-flowers /opt/program/datasets/trees-flowers

RUN mv /opt/program/train.py /opt/program/train
RUN mv /opt/program/test.py /opt/program/serve

RUN chmod +x /opt/program/train

RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl --upgrade && \
    pip3 install torchvision --upgrade
    
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program