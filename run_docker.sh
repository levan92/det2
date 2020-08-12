export WORKSPACE=$HOME/Workspace
export DATA=/media/dh/HDD

xhost +local:docker
docker run -it -v $WORKSPACE:$WORKSPACE \
-v $DATA:$DATA \
--net=host --ipc host --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --env="QT_X11_NO_MITSHM=1" \
--device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1 \
levan92/det2