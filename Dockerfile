FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --ignore-installed -r requirements.docker.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mkl-service
WORKDIR /BirSwinT