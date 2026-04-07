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
        backend: str = 'openvino',
        device: str = 'auto',
        dst_dir=None
    ):
        super().__init__(
            onnx_model,
            model_input_size,
            backend=backend,
            device=device,
            dst_dir=dst_dir
        )
        self.mode = mode
        self.nms_thr = nms_thr
        self.score_thr = score_thr

    def __call__(self, image: np.ndarray):
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)

        if isinstance(outputs, list):
#           print("YOLO11 raw output shapes:", [np.asarray(out).shape for out in outputs])
            outputs = max(outputs, key=lambda x: np.asarray(x).size)

        outputs = np.asarray(outputs)
#        print("YOLO11 selected output shape:", outputs.shape)

        return self.postprocess(outputs, ratio)

    def preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        target_h, target_w = self.model_input_size[:2]

        ratio = min(target_h / h, target_w / w)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return padded, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        outputs = np.asarray(outputs)

        # exported with NMS: (1, N, 6) => x1,y1,x2,y2,score,class
        if outputs.ndim == 3 and outputs.shape[0] == 1 and outputs.shape[-1] == 6:
            preds = outputs[0]
            final_boxes = preds[:, :4] / ratio
            final_scores = preds[:, 4]
            final_cls_inds = preds[:, 5].astype(int)

            score_keep = final_scores > self.score_thr
            final_boxes = final_boxes[score_keep]
            final_scores = final_scores[score_keep]
            final_cls_inds = final_cls_inds[score_keep]

        # raw output: (1, 84, N) benzeri
        elif outputs.ndim == 3 and outputs.shape[0] == 1 and outputs.shape[1] == 84:
            preds = outputs[0].transpose(1, 0)  # (N, C)
            boxes = preds[:, :4]
            cls_scores = preds[:, 4:]

            cls_inds = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(len(cls_scores)), cls_inds]

            score_keep = scores > self.score_thr
            boxes = boxes[score_keep]
            scores = scores[score_keep]
            cls_inds = cls_inds[score_keep]

            if len(boxes) == 0:
                if self.mode == 'multiclass':
                    return (
                        np.empty((0, 4), dtype=np.float32),
                        np.empty((0,), dtype=np.int32),
                    )
                return np.empty((0, 4), dtype=np.float32)

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

            if dets is None:
                if self.mode == 'multiclass':
                    return (
                        np.empty((0, 4), dtype=np.float32),
                        np.empty((0,), dtype=np.int32),
                    )
                return np.empty((0, 4), dtype=np.float32)

            final_boxes = dets[:, :4]
            final_scores = dets[:, 4]
            final_cls_inds = dets[:, 5].astype(int)

        else:
            raise ValueError(f"Unexpected YOLO11 output shape: {outputs.shape}")

#        print("final_boxes:", final_boxes.shape if len(final_boxes) else final_boxes)
#        print(
#            "final_cls_inds:",
#            final_cls_inds[:10] if len(final_cls_inds) else final_cls_inds
#        )

        if self.mode == 'multiclass':
            return final_boxes, final_cls_inds

        if self.mode == 'human':
            person_keep = final_cls_inds == 0
#            print("person count:", int(np.sum(person_keep)))
            return final_boxes[person_keep]

        raise NotImplementedError(f"Unsupported mode: {self.mode}")