# 神奇的xml文件用于解决连不上G1 low-state问题
export CYCLONEDDS_URI=file://$(pwd)/cyclonedds.xml

# 手柄需插在台式服务器，而不是远端笔记本
export SDL_JOYSTICK_DEVICE=/dev/input/js0

# 启动命令，请自己修改run_pipeline.py内的参数
python scripts/run_pipeline.py