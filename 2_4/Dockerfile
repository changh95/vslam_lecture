FROM slam:latest

RUN cd vslam_lecture/2_4 &&\
    mkdir build && cd build && \
    cmake -GNinja ../ && \
    ninja -j4