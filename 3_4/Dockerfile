FROM slam:latest

RUN git clone https://github.com/dorian3d/DBoW2.git &&\
    cd DBoW2 &&\
    mkdir build && cd build &&\
    cmake -GNinja .. &&\
    ninja &&\
    ninja install &&\
    cd ../../

RUN cd vslam_lecture/3_4 &&\
    mkdir build && cd build && \
    cmake -GNinja ../ && \
    ninja