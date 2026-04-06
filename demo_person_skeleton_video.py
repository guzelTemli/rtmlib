import cv2
from person_skeleton_detection import PersonSkeletonDetection


def main():
    # --------------------------------------------------
    # VIDEO YOLU
    # --------------------------------------------------
    video_path = r"video\\12740312_3840_2160_24fps.mp4"

    # --------------------------------------------------
    # MODEL YOLLARI
    # Kendi yollarına göre değiştir
    # --------------------------------------------------
    det_model_path = r"C:\\Users\\gzltm\\source\\GitHub\\rtmlib\\rtmlib\\weights\\yolo11n.onnx"
    pose_model_path = r"C:\\Users\\gzltm\\source\\GitHub\\rtmlib\\rtmlib\\weights\\rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx"

    # --------------------------------------------------
    # BACKEND / DEVICE AYARLARI
    # --------------------------------------------------
    backend = "openvino"   # opencv, onnxruntime, openvino
    device = "cpu"         # cpu veya cuda
    mode = "balanced"
    det_frequency = 7
    tracking = True
    to_openpose = False
    kpt_thr = 0.5

    # --------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------
    output_path = r"output_person_skeleton.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video açılamadı: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    if input_fps <= 0:
        input_fps = 25.0

    print("=== AYARLAR ===")
    print(f"Video             : {video_path}")
    print(f"Detector model    : {det_model_path}")
    print(f"Pose model        : {pose_model_path}")
    print(f"Backend           : {backend}")
    print(f"Device            : {device}")
    print(f"Mode              : {mode}")
    print(f"det_frequency     : {det_frequency}")
    print(f"Tracking          : {tracking}")
    print(f"Output            : {output_path}")
    print(f"Input size        : {width}x{height}")
    print(f"Input FPS         : {input_fps:.2f}")

    detector = PersonSkeletonDetection(
        det_model_path=det_model_path,
        pose_model_path=pose_model_path,
        backend=backend,
        device=device,
        mode=mode,
        det_frequency=det_frequency,
        tracking=tracking,
        to_openpose=to_openpose,
        kpt_thr=kpt_thr,
        show_fps=True,
    )

    writer = detector.create_video_writer(output_path, width, height, input_fps)

    window_name = "Person Skeleton Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # İstersen daha küçük açılması için:
    preview_width = 1280
    preview_height = 720
    cv2.resizeWindow(window_name, preview_width, preview_height)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        keypoints, scores = detector.process_frame(frame)
        vis_frame = detector.draw(frame, keypoints, scores)

        writer.write(vis_frame)
        cv2.imshow(window_name, vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\nTamamlandı. Toplam frame: {frame_count}")
    print(f"Kayıt edilen çıktı: {output_path}")


if __name__ == "__main__":
    main()