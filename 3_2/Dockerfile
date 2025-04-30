FROM slam:latest

RUN cd vslam_lecture/ &&\
    git pull && cd 3_2 && mkdir build && cd build && \
    cmake -GNinja ../ && \
    ninja