FROM slam:latest

MAINTAINER changh95
ARG DEBIAN_FRONTEND=noninteractive

RUN cd vslam_lecture/1_3 &&\
    git pull &&\
    mkdir build && cd build &&\
    cmake -GNinja .. &&\
    ninja -j4