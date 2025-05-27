# Local feature extraction & matching, using OpenCV library

## How to build

```
docker build . -t slam:3_2
```

## How to run

```
xhost +local:root

docker run -it \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --privileged \
  --env="XAUTHORITY=/root/.Xauthority" \
  slam:3_2

```

## How to run examples

```
cd examples
./detect_feature
```