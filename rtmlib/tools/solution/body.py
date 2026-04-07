'''
Example:

import cv2

from rtmlib import Body, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.mp4')

openpose_skeleton = True  # True for openpose-style, False for mmpose-style

body = Body(to_openpose=openpose_skeleton,
                      backend=backend,
                      device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = body(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)

'''
import numpy as np


class Body:
    MODE = {
        'performance': {
            'det':
            'rtmlib\\weights\\openvino_format\\yolo11n_openvino_model.xml',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'rtmlib\\weights\\openvino_format\\rtmposepose_openvino_model.xml',  # noqa
            'pose_input_size': (288, 384),
            'dst_dir': "rtmlib/weights"
        },
        'lightweight': {
            'det':
            'rtmlib\\weights\\openvino_format\\yolo11n_openvino_model.xml',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'rtmlib\\weights\\openvino_format\\rtmposepose_openvino_model.xml',  # noqa
            'pose_input_size': (192, 256),
            'dst_dir': "rtmlib/weights"
        },
        'balanced': {
            'det':
            'rtmlib\\weights\\openvino_format\\yolo11n_openvino_model.xml',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'rtmlib\\weights\\openvino_format\\rtmposepose_openvino_model.xml',  # noqa
            'pose_input_size': (192, 256),
            'dst_dir': "rtmlib/weights"
        }
    }

    RTMO_MODE = {
        'performance': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip',  # noqa
            'pose_input_size': (640, 640),
            'dst_dir': "rtmlib/weights"
        },
        'lightweight': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip',  # noqa
            'pose_input_size': (640, 640),
            'dst_dir': "rtmlib/weights"
        },
        'balanced': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip',  # noqa
            'pose_input_size': (640, 640),
            'dst_dir': "rtmlib/weights"
        }
    }

    def __init__(self,
                 det: str = None,
                 det_input_size: tuple = (640, 640),
                 pose: str = None,
                 pose_input_size: tuple = (288, 384),
                 mode: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 dst_dir=None):

        if pose is not None and 'rtmo' in pose:
            from .. import RTMO

            self.one_stage = True

            pose = self.RTMO_MODE[mode]['pose']
            pose_input_size = self.RTMO_MODE[mode]['pose_input_size']
            self.pose_model = RTMO(pose,
                                   model_input_size=pose_input_size,
                                   to_openpose=to_openpose,
                                   backend=backend,
                                   device=device,
                                   dst_dir=dst_dir)
        else:
            from .. import YOLO11, RTMPose

            self.one_stage = False

            if pose is None:
                pose = self.MODE[mode]['pose']
                pose_input_size = self.MODE[mode]['pose_input_size']

            if det is None:
                det = self.MODE[mode]['det']
                det_input_size = self.MODE[mode]['det_input_size']

            self.det_model = YOLO11(det,
                                   model_input_size=det_input_size,
                                   backend=backend,
                                   device=device,
                                   dst_dir=dst_dir)
            self.pose_model = RTMPose(pose,
                                      model_input_size=pose_input_size,
                                      to_openpose=to_openpose,
                                      backend=backend,
                                      device=device,
                                      dst_dir=dst_dir)

    def __call__(self, image: np.ndarray):
        if self.one_stage:
            keypoints, scores = self.pose_model(image)
        else:
            bboxes = self.det_model(image)
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores
