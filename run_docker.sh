xhost +local:docker
docker run -it -v $HOME/Workspace/:$HOME/Workspace \
-v /media/dh/HDD/:/media/dh/HDD/ \
--net=host --ipc host --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --env="QT_X11_NO_MITSHM=1" \
--device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1 \
levan92/cv-suite