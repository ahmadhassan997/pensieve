FROM ubuntu:18.04

RUN apt-get update && apt-get install -y debconf-utils \
 && echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get install -y \
    curl ca-certificates sudo git bzip2 libx11-6 gcc cmake \
    libboost-all-dev dialog apt-utils

