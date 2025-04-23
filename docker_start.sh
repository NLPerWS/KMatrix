# !/bin/bash

# docker pull leap233/kmatrix:v1

# 打包镜像
# docker save -o kmatrix.tar leap233/kmatrix:v1
# 解压镜像
# 检查镜像是否存在

IMAGE_NAME="leap233/kmatrix:v1"

if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
  echo "镜像 ${IMAGE_NAME} 不存在。"
  echo "正在加载镜像..."
  # 加载镜像
  docker load -i kmatrix.tar
  if [ $? -eq 0 ]; then
    echo "镜像加载成功。"
  else
    echo "镜像加载失败，请检查 kmatrix.tar 文件是否存在或格式是否正确。"
  fi
else
  echo "镜像 ${IMAGE_NAME} 已存在。"
fi


if [ "$(docker ps -aq -f name=kmatrix_run)" ]; then
    echo "Stopping and removing existing container kmatrix_run..."
    docker rm -f kmatrix_run
else
    echo "No existing container kmatrix_run found."
fi

# (Optional) Add the mapping path corresponding to the model in the startup command, so that you do not need to re-download the model when restarting the container (for the KMatrix model path, refer to root_config.py)
# For example:
# -v /netcache/huggingface/Baichuan2-7B-Chat/:/app/KMatrix/dir_model/generator/Baichuan2-13B-Chat \


# Note that the port number should not conflict with other services.
docker run -idt --gpus all \
    -p 10005:10005 \
    -p 10006:10006 \
    -v $(pwd):/app/KMatrix/ \
    --name kmatrix_run leap233/kmatrix:v1 \
    /bin/bash '/app/KMatrix/startup.sh'


# Enter the container for debugging
# docker exec -it kmatrix_run /bin/bash