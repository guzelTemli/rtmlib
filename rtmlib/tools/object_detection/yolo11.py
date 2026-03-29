from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool
from .post_processings import multiclass_nms


class YOLO11(BaseTool):
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple = (640, 640),
        mode: str = 'human',
        nms_thr: float = 0.45,
        score_thr: float = 0.25,
        backend: str = 'onnxruntime',
        device: str = 'cpu',
    ):
        super().__init__(
            onnx_model,
            model_input_size,
            backend=backend,
            device=device
        )
        self.mode = mode
        self.nms_thr = nms_thr
        self.score_thr = score_thr

    def __call__(self, image: np.ndarray):
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)[0]
        results = self.postprocess(outputs, ratio)
        return results

    def preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        target_h, target_w = self.model_input_size[:2]

        ratio = min(target_h / h, target_w / w)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0

        padded = np.transpose(padded, (2, 0, 1))
        padded = np.expand_dims(padded, axis=0)

        return padded, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        outputs = np.array(outputs)

        # raw detect output: (1, 84, N)
        if outputs.ndim == 3 and outputs.shape[1] >= 5 and outputs.shape[1] < outputs.shape[2]:
            preds = outputs[0].transpose(1, 0)  # (N, 84)
            boxes = preds[:, :4]
            cls_scores = preds[:, 4:]

            cls_inds = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(len(cls_scores)), cls_inds]

            keep = scores > self.score_thr
            boxes = boxes[keep]
            scores = scores[keep]
            cls_inds = cls_inds[keep]

            # xywh -> xyxy
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            boxes_xyxy /= ratio

            num_classes = len(self.COCO_CLASSES)
            score_matrix = np.zeros((len(scores), num_classes), dtype=np.float32)
            score_matrix[np.arange(len(scores)), cls_inds] = scores

            dets, _ = multiclass_nms(
                boxes_xyxy,
                score_matrix,
                nms_thr=self.nms_thr,
                score_thr=self.score_thr
            )

            if dets is not None:
                final_boxes = dets[:, :4]
                final_scores = dets[:, 4]
                final_cls_inds = dets[:, 5].astype(int)
            else:
                final_boxes = np.empty((0, 4), dtype=np.float32)
                final_scores = np.empty((0,), dtype=np.float32)
                final_cls_inds = np.empty((0,), dtype=np.int32)

        # exported with nms: (1, N, 6)
        elif outputs.ndim == 3 and outputs.shape[-1] >= 6:
            preds = outputs[0]
            final_boxes = preds[:, :4] / ratio
            final_scores = preds[:, 4]
            final_cls_inds = preds[:, 5].astype(int)

            keep = final_scores > self.score_thr
            final_boxes = final_boxes[keep]
            final_scores = final_scores[keep]
            final_cls_inds = final_cls_inds[keep]
        else:
            raise ValueError(f"Unexpected YOLO11 output shape: {outputs.shape}")

        if self.mode == 'multiclass':
            return final_boxes, final_cls_inds
        elif self.mode == 'human':
            print("final_cls_inds:", final_cls_inds)
            print("final_scores:", final_scores)
            print("final_boxes:", final_boxes)
            return final_boxes[keep]
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")