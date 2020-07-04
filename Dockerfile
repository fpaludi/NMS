FROM python:3.7
COPY requirements.txt /model/
WORKDIR /model
RUN pip install -r requirements.txt
COPY model.py /model/
ADD ./protos /model/protos
ADD ./utils /model/utils
ADD ./faster_rcnn_inception_v2_coco_2018_01_28 /model/faster_rcnn_inception_v2_coco_2018_01_28
#ENV NMS_TH=0.4
#ENV SCORE_TH=0.5
#CMD ["python", "model.py"]
