# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import TestRequirements
from tracking.detectors import get_yolo_inferer

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git',))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box
import threading
from sklearn.linear_model import LinearRegression


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(yolo, args, stream):
    results = yolo.track(
        source=stream,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=False,  # args.show_conf,
        save_txt=args.save_txt,
        show_labels=False,  # args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    # store custom args in predictor
    yolo.predictor.custom_args = args
    trajectories = {}
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 25.0, (1280, 720))
    frame_box = np.zeros((720, 1280), dtype=np.uint8)
    # idx = 1
    up = 0
    down = 0
    for r in results:
        middle_line_y = r.orig_img.shape[0] // 2
        current_box = np.zeros((720, 1280), dtype=np.uint8)
        for i in range(len(r.boxes)):
            tracking_id = r.boxes[i].id.item()
            box_center_y = ((r.boxes[i].xyxy[:, 1] + r.boxes[i].xyxy[:, 3]) / 2).item()
            box_center_x = ((r.boxes[i].xyxy[:, 0] + r.boxes[i].xyxy[:, 2]) / 2).item()
            # iterate through coordinates of the bounding box
            for j in range(int(r.boxes[i].xyxy[:, 0].item()), int(r.boxes[i].xyxy[:, 2].item())):
                for k in range(int(r.boxes[i].xyxy[:, 1].item()), int(r.boxes[i].xyxy[:, 3].item())):
                    frame_box[k, j] = 1
                    current_box[k, j] = 1
            if tracking_id not in trajectories:
                trajectories[tracking_id] = np.array([box_center_x, box_center_y]).reshape(-1, 2)
            else:
                prevy = trajectories[tracking_id][trajectories[tracking_id].shape[0] - 1][1]
                if prevy < middle_line_y and box_center_y >= middle_line_y:
                    if box_center_y - prevy >= 1:
                        print(f"#{tracking_id} person crossed the line DOWN")
                        down += 1
                elif prevy >= middle_line_y and box_center_y < middle_line_y:
                    if prevy - box_center_y >= 1:
                        print(f"#{tracking_id} person crossed the line UP")
                        up += 1
                trajectories[tracking_id] = np.vstack(
                    [trajectories[tracking_id], np.array([box_center_x, box_center_y])])
            directionEstimator = LinearRegression()
            X_train = np.arange(1, trajectories[tracking_id].shape[0] + 1).reshape(-1, 1)
            directionEstimator.fit(X_train,
                                   trajectories[tracking_id].reshape(-1, 2))
            estimatedDirection = directionEstimator.predict(
                np.arange(trajectories[tracking_id].shape[0] + 1, trajectories[tracking_id].shape[0] + 100).reshape(-1,
                                                                                                                    1))
            # draw connected lines on r.orig_img based on estimatedDirection
            for i in range(estimatedDirection.shape[0] - 1):
                cv2.line(r.orig_img, (int(estimatedDirection[i][0]), int(estimatedDirection[i][1])),
                         (int(estimatedDirection[i + 1][0]), int(estimatedDirection[i + 1][1])), (0, 255, 0), 2)
            # print(f"#{tracking_id} estimated direction: {estimatedDirection}")

        img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)
        # draw horizontal middle line on r.orig_img
        cv2.line(img, (0, middle_line_y), (img.shape[1], middle_line_y), (0, 255, 0), 2)
        position = (20, img.shape[0] - 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        crowd_density = int(np.sum(current_box) / np.sum(frame_box)) * 100
        cv2.putText(img, f"Count: {len(r.boxes)}, up: {up}, down: {down}, crowd:{crowd_density}%", position, font,
                    font_scale, color, thickness)
        out.write(img)
        # cv2.imwrite(f'images/output_{idx}.jpg', img)
        # idx+=1

        # if args.show is True:
        #     cv2.imshow('BoxMOT', img)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord(' ') or key == ord('q'):
        #         break
    out.release()
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    # Load the models
    model1 = YOLO('yolov8n.pt')
    model2 = YOLO('yolov8n.pt')

    # Define the video files for the trackers
    stream1 = 'track.mp4'
    # stream1 = 'rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/2/media.smp'  # Path to video file, 0 for webcam
    stream2 = 'rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/0/media.smp'  # Path to video file, 0 for webcam, 1 for external camera

    # Create the tracker threads
    tracker_thread1 = threading.Thread(target=run, args=(model1, opt, stream1), daemon=True)
    tracker_thread2 = threading.Thread(target=run, args=(model2, opt, stream2), daemon=True)

    # Start the tracker threads
    tracker_thread1.start()
    # tracker_thread2.start()

    # Wait for the tracker threads to finish
    tracker_thread1.join()
    # tracker_thread2.join()
    # run(opt)
