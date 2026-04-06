import time
from pathlib import Path

import cv2
from rtmlib import PoseTracker, Wholebody, draw_skeleton


class PersonSkeletonDetection:
    def __init__(
        self,
        det_model_path: str,
        pose_model_path: str,
        backend: str = "openvino",
        device: str = "cpu",
        mode: str = "balanced",
        det_frequency: int = 10,
        tracking: bool = True,
        to_openpose: bool = False,
        kpt_thr: float = 0.5,
        show_fps: bool = True,
    ):
        self.det_model_path = det_model_path
        self.pose_model_path = pose_model_path
        self.backend = backend
        self.device = device
        self.mode = mode
        self.det_frequency = det_frequency
        self.tracking = tracking
        self.to_openpose = to_openpose
        self.kpt_thr = kpt_thr
        self.show_fps = show_fps

        self.pose_tracker = PoseTracker(
            Wholebody,
            mode=self.mode,
            backend=self.backend,
            device=self.device,
            det_frequency=self.det_frequency,
            tracking=self.tracking,
            to_openpose=self.to_openpose,
        )

        self.prev_time = None
        self.fps = 0.0

    def process_frame(self, frame):
        keypoints, scores = self.pose_tracker(frame)
        return keypoints, scores

    def draw(self, frame, keypoints, scores):
        output = frame.copy()

        output = draw_skeleton(
            output,
            keypoints,
            scores,
            kpt_thr=self.kpt_thr,
        )

        if self.show_fps:
            current_time = time.time()
            if self.prev_time is not None:
                dt = current_time - self.prev_time
                if dt > 0:
                    self.fps = 1.0 / dt

            self.prev_time = current_time

            cv2.putText(
                output,
                f"FPS: {self.fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return output

    @staticmethod
    def create_video_writer(output_path: str, width: int, height: int, fps: float):
        output_path = str(Path(output_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return writer