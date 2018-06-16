FROM nvidia/cuda:9.0-runtime

RUN apt-get update && \
    apt-get -y install build-essential python-dev python3-dev python3-pip python-imaging wget curl

RUN apt-get -y install git

RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl --upgrade && \
    pip3 install torchvision --upgrade

RUN apt-get install -y git
RUN git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git /tmp/pytorch

RUN pip3 install dominate
RUN pip3 install scipy
RUN pip3 install visdom

RUN mkdir /opt/program
RUN cd /opt/program

RUN mv /tmp/pytorch/* /opt/program/

COPY train.py /opt/program/train
COPY test.py  /opt/program/serve

RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve

RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program