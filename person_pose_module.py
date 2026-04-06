import cv2
import numpy as np
from collections import deque
from rtmlib import Custom, draw_skeleton


class PersonPoseModule:
    def __init__(
        self,
        det_model_path,
        pose_model_path,
        backend="openvino",
        device="cpu",
        det_input_size=(640, 640),
        pose_input_size=(192, 256),
        kpt_thr=0.4,
        det_interval=10,
        to_openpose=False,
    ):
        self.det_model_path = det_model_path
        self.pose_model_path = pose_model_path
        self.backend = backend
        self.device = device
        self.det_input_size = det_input_size
        self.pose_input_size = pose_input_size
        self.kpt_thr = kpt_thr
        self.det_interval = det_interval
        self.to_openpose = to_openpose

        self.custom = Custom(
            to_openpose=self.to_openpose,
            det_class='YOLO11',
            det=self.det_model_path,
            det_input_size=self.det_input_size,
            pose_class='RTMPose',
            pose=self.pose_model_path,
            pose_input_size=self.pose_input_size,
            backend=self.backend,
            device=self.device
        )

        self.custom.det_model.score_thr = 0.5

        self.frame_index = 0
        self.last_bboxes = []
        self.last_keypoints = []
        self.last_scores = []

        self.fps_buf = deque(maxlen=30)

    def process_frame(self, frame):
        self.frame_index += 1

        if self.frame_index % self.det_interval == 1:
            bboxes = self.custom.det_model(frame)

            if len(bboxes) > 0:
                keypoints, scores = self.custom.pose_model(frame, bboxes=bboxes)
            else:
                bboxes, keypoints, scores = [], [], []

            self.last_bboxes = bboxes
            self.last_keypoints = keypoints
            self.last_scores = scores

        return self.last_bboxes, self.last_keypoints, self.last_scores

    def draw(self, frame, bboxes, keypoints, scores, inference_ms=None):
        display = frame.copy()

        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"Person {i}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if len(keypoints) > 0:
            display = draw_skeleton(
                display,
                np.array(keypoints),
                scores,
                kpt_thr=self.kpt_thr
            )

        if inference_ms is not None and inference_ms > 0:
            fps = 1000.0 / inference_ms
            self.fps_buf.append(fps)
            avg_fps = sum(self.fps_buf) / len(self.fps_buf)

            cv2.rectangle(display, (0, 0), (280, 90), (0, 0, 0), -1)
            cv2.putText(display, f"FPS   : {avg_fps:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"ms    : {inference_ms:.1f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Person: {len(keypoints)}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return display