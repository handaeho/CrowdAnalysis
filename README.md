# Crowd analsys

- Human detection and tracking
- Human counting based on the line
- Multi camera processing

1. Clone the project repository

2. RUN Dockerfile to create container

3. RUN poetry shell

4. RUN python track_stream.py --conf 0.2 --classes 0 --iou 0.3 --per-class --agnostic-nms 

    (argument detail info is in track_stream.py)
