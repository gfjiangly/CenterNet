ARG PYTORCH="0.4.1"
ARG CUDA="9"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install COCOAPI
#RUN conda install cython -y && conda clean --all
RUN pip install cython
RUN cd /root
RUN git clone https://github.com/cocodataset/cocoapi.git /root/cocoapi
#RUN cd /root/cocoapi/PythonAPI
#RUN make
#RUN python setup.py install --user

# Install CenterNet
RUN git clone https://github.com/xingyizhou/CenterNet.git /CenterNet
WORKDIR /CenterNet
RUN pip install -r requirements.txt
#RUN cd /CenterNet/src/lib/models/networks/DCNv2
#RUN ./make.sh
#RUN cd /CenterNet/src/lib/external
#RUN make

