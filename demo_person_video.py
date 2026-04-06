import cv2
import time
from person_pose_module import PersonPoseModule


def main():
    video_path = r"video\\12740312_3840_2160_24fps.mp4"

    det_model_path = r"C:\\Users\\gzltm\\source\\GitHub\\rtmlib\\rtmlib\\weights\\yolo11n.onnx"
    pose_model_path = r"C:\\Users\\gzltm\\source\\GitHub\\rtmlib\\rtmlib\\weights\\rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx"

    backend = "openvino"
    device = "cpu"
    output_path = "output_person_skeleton_fast.mp4"

    inf_w = 960
    inf_h = 540

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video açılamadı: {video_path}")
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, input_fps, (inf_w, inf_h))

    detector = PersonPoseModule(
        det_model_path=det_model_path,
        pose_model_path=pose_model_path,
        backend=backend,
        device=device,
        det_input_size=(640, 640),
        pose_input_size=(192, 256),
        kpt_thr=0.4,
        det_interval=10,
        to_openpose=False,
    )

    print("=== HIZLI DEMO AYARLARI ===")
    print(f"Video          : {video_path}")
    print(f"Output         : {output_path}")
    print(f"Inference size : {inf_w}x{inf_h}")
    print(f"Backend        : {backend}")
    print(f"Device         : {device}")
    print(f"det_interval   : 10")

    cv2.namedWindow("Person Skeleton Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Person Skeleton Demo", 960, 540)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video sona erdi.")
            break

        frame = cv2.resize(frame, (inf_w, inf_h))

        t0 = time.time()
        bboxes, keypoints, scores = detector.process_frame(frame)
        t1 = time.time()

        inference_ms = (t1 - t0) * 1000.0

        display = detector.draw(frame, bboxes, keypoints, scores, inference_ms)

        writer.write(display)
        cv2.imshow("Person Skeleton Demo", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Kayıt tamamlandı: {output_path}")


if __name__ == "__main__":
    main()