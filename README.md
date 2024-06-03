1. Get a data set
2. use only 0.1 % to test cleaning  and so on
3. Train 10% of data set to get the correct models + validations
4. train on real data set (from public source)
5. If models proven successful, scrape data from the web


# Env set up
connect to another computer:
`ssh hoang@192.168.1.102` => password 2910
`cd ~/home/hoang/Documents/work/ray-worker-node`
`source myenv/bin/activate`

`sudo docker build -t my-python-app . && sudo docker run --gpus all -it --rm my-python-app`

`sudo docker system prune`

command line to track resources `htop` + `nvtop`



# To install tensorflow with GPU
1. Make sure to have nvidia graphic card
`lspci | grep -i nvidia`
2. install the latest nvidia-driver, search on the nvidia website
`sudo apt install nvidia-driver-xxx`
To verifiy the driver `nvidia-smi` 
3. Install docker, make sure there's only one docker `context` in the system. DO NOT install docker desktop. it will created 2 contexts.
```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo service docker start
```
make sure that the current context is `default` verified by `sudo docker context list`

4. Install docker container toolkit from nvidia

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list


sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker
```
5. Test the installation: If you see the tensor, it's working
`docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`



6. For distributed training
Head node
`sudo docker stop worker-0 | true && sudo docker rm worker-0 | true && sudo docker rmi --force my_tensorflow_app && sudo docker build -t my_tensorflow_app . && sudo docker run --gpus all -p 2222:2222 -e TF_CONFIG='{"cluster": {"worker": ["192.168.1.101:2222", "192.168.1.102:2222"]}, "task": {"type": "worker", "index": 0}}' --name worker-0  my_tensorflow_app`



Worker Node
`cd /home/hoang/Documents/work/popo24_ML_engine`

`sudo docker stop worker-0 | true && sudo docker rm worker-0 | true && sudo docker rmi --force my_tensorflow_app && sudo docker build -t my_tensorflow_app . && sudo docker run --gpus all -p 2222:2222 -e TF_CONFIG='{"cluster": {"worker": ["192.168.1.101:2222", "192.168.1.102:2222"]}, "task": {"type": "worker", "index": 1}}' --name worker-0  my_tensorflow_app`