# !/bin/bash

if [ "$(docker ps -aq -f name=kmatrix_run)" ]; then
    echo "Stopping and removing existing container kmatrix_run..."
    docker rm -f kmatrix_run
else
    echo "No existing container kmatrix_run found."
fi

# (Optional) Add the mapping path corresponding to the model in the startup command, so that you do not need to re-download the model when restarting the container (for the KMatrix model path, refer to root_config.py)
# For example:
# -v /netcache/huggingface/Llama-2-7b-chat-hf/:/app/KMatrix/dir_model/generator/Llama-2-7b-chat-hf \


# Note that the port number should not conflict with other services.
docker run -idt --gpus all \
    -p 8000:8000 \
    -p 8002:8002 \
    -v $(pwd):/app/KMatrix/ \
    --name kmatrix_run kmatrix:v1 \
    /bin/bash '/app/KMatrix/startup.sh'


# Enter the container for debugging
# docker exec -it kmatrix_run /bin/bash