FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# TODO: Add --no-install-recommends to reduce image size (requires going step-by-step)
RUN apt update && \
  apt upgrade -y && \
  apt install -y \
    gcc \
    g++ \
    cmake \
    mold \
    git \
    clang-format-15 \
    cmake-format \
    unzip \
    python3-pip \
    python3-venv \
    libtbb-dev \
    libeigen3-dev \
    libglew-dev \
    ccache \
    libgl1-mesa-dev \
    libjpeg-dev \
    libpng-dev \
    liblz4-dev \
    libbz2-dev \
    libboost-regex-dev \
    libboost-filesystem-dev \
    libboost-date-time-dev \
    libboost-program-options-dev \
    libgtest-dev \
    libopencv-dev \
    libfmt-dev \
    ninja-build \
    libepoxy-dev \
    curl && \
  export CLANGD_URL=$(curl -s https://api.github.com/repos/clangd/clangd/releases/latest | grep -oP "\"browser_download_url.*clangd-linux-(\d|\.)+\.zip\"" | cut -d ":" -f 2,3 | tr -d \") && \
    echo "Setting up clangd-tidy" && \
    echo "CLANGD_URL=" $CLANGD_URL && \
    curl -L --silent --show-error --fail $CLANGD_URL -o clangd-linux.zip && \
    unzip -q clangd-linux.zip && \
    rm clangd-linux.zip && \
    mv clangd* /clangd && \
    python3 -m venv /venv && \
    . /venv/bin/activate && \
    pip install clangd-tidy
RUN apt-get install -y apt-transport-https ca-certificates gnupg

# Download dataset
RUN apt-get install -y wget

# Valve Index
RUN wget https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MI_valve_index/MIO_others/MIO05_inspect_easy.zip
RUN wget https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MI_valve_index/MIP_playing/MIPB_beat_saber/MIPB07_beatsaber_fitbeat_expertplus_2.zip
RUN unzip MIPB07_beatsaber_fitbeat_expertplus_2.zip

# Samsung Odyssey Plus
RUN wget https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MO_odyssey_plus/MOO_others/MOO07_mapping_easy.zip
RUN unzip MOO07_mapping_easy.zip

# Install Basalt
RUN git clone --recursive https://github.com/changh95/vslam_lecture.git
WORKDIR vslam_lecture/basalt_monado
RUN apt-get install -y sudo
RUN ./scripts/install_deps.sh
RUN cmake --preset development
RUN sudo cmake --build build --target install

# Build Command:
# docker build . -t basalt

# Run Command:
# docker run -it --privileged --net=host --ipc=host -e "DISPLAY=$DISPLAY" -e "QT_X11_NO_MITSHM=1" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -e "XAUTHORITY=$XAUTH" --gpus all basalt

#docker run -it --rm \
#    --gpus all \
#    --net=host \
#    --ipc=host \
#    --env DISPLAY=$DISPLAY \
#    --env QT_X11_NO_MITSHM=1 \
#    --env XAUTHORITY=$HOME/.Xauthority \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -v $HOME/.Xauthority:$HOME/.Xauthority \
#    --privileged \
#    basalt

# Inside docker, run command:

# (Samsung Odyssey Plus)
# basalt_vio --show-gui 1 --dataset-path ../MOO09_short_1_updown/ --dataset-type euroc --cam-calib ./data/msd/msdmo_calib.json --config-path ./data/msd/msdmo_config.json

