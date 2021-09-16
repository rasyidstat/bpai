FROM ubuntu:18.04

ENV PATH /root/miniconda3/bin:$PATH
ARG CONDA=Miniconda3-py38_4.9.2-Linux-x86_64.sh

RUN apt update \ 
    && apt install -y htop wget build-essential make pkg-config

RUN wget https://repo.anaconda.com/miniconda/${CONDA} \
    && mkdir root/.conda \
    && sh ${CONDA} -b \
    && rm -f ${CONDA}

WORKDIR /work/

COPY ./requirements.txt /work/requirements.txt
RUN pip install -r requirements.txt

COPY . /work/