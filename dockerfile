FROM python:3.10
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
#RUN pip install --user tensorboard cmake onnx   # cmake from apt-get is too old
#RUN pip install --user torch==1.11 torchvision==0.12.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html

#RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
#ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
#ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
#ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

#RUN git clone https://github.com/mikel-brostrom/yolo_tracking.git yolo_tracking
WORKDIR /app

# Set a fixed model cache directory.
#ENV FVCORE_CACHE="/tmp"
#WORKDIR /app
#COPY track_stream.py .
#COPY input input
COPY . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 vim -y
#RUN pip install opencv-python
WORKDIR /app/yolo_tracking
RUN pip install poetry
RUN poetry install --with yolo  # installed boxmot + yolo dependencies
#RUN poetry shell  # activates the newly created environment with the installed dependencies
#RUN pip install boxmot
# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl`
