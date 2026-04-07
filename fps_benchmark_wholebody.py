import time
import cv2
import numpy as np

from rtmlib import PoseTracker, Wholebody, draw_skeleton

# =========================
# AYARLAR
# =========================
device = 'cpu'
backend = 'openvino'   # opencv, onnxruntime, openvino
video_path = "C:\\Users\\gzltm\\source\\GitHub\\rtmlib\\video\\tek_adam.mp4"   # video dosya yolun
openpose_skeleton = False

mode = 'balanced'      # balanced, performance, lightweight
det_frequency = 7      # detector kac framede bir calissin
warmup_frames = 30     # ilk frameleri olcme, sistem isinma yapsin
test_frames = 200      # kac frame uzerinden ortalama alinacak
show_window = True     # ekranda gosterilsin mi

# =========================
# VIDEO AC
# =========================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Video acilamadi: {video_path}")
    raise SystemExit

# =========================
# MODEL KUR
# =========================
wholebody = PoseTracker(
    Wholebody,
    det_frequency=det_frequency,
    to_openpose=openpose_skeleton,
    mode=mode,
    backend=backend,
    device=device
)

# =========================
# OLÇUM LISTELERI
# =========================
infer_times = []   # sadece model suresi
total_times = []   # frame+model+draw+imshow toplam sure
frame_idx = 0

# =========================
# ANA DONGU
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # tum pipeline suresi
    total_start = time.perf_counter()

    # sadece model suresi
    infer_start = time.perf_counter()
    keypoints, scores = wholebody(frame)
    infer_end = time.perf_counter()

    # skeleton ciz
    img_show = frame.copy()
    img_show = draw_skeleton(
        img_show,
        keypoints,
        scores,
        openpose_skeleton=openpose_skeleton,
        kpt_thr=0.4
    )

    # anlik fps hesapla
    current_infer_time = infer_end - infer_start
    current_infer_fps = 1.0 / current_infer_time if current_infer_time > 0 else 0.0

    # ekrana yaz
    cv2.putText(
        img_show,
        f"Infer FPS: {current_infer_fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    h, w = img_show.shape[:2]
    max_width = 1280
    if w > max_width:
        scale = max_width / w
        img_show = cv2.resize(img_show, (None, None), fx=scale, fy=scale)

    total_end = time.perf_counter()

    # warmup bittikten sonra sureleri kaydet
    if frame_idx > warmup_frames:
        infer_times.append(infer_end - infer_start)
        total_times.append(total_end - total_start)

    # ekranda goster
    if show_window:
        cv2.imshow("Wholebody FPS Test", img_show)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break

    # yeterli frame toplandiysa cik
    if len(infer_times) >= test_frames:
        break

# =========================
# TEMIZLIK
# =========================
cap.release()
cv2.destroyAllWindows()

# =========================
# RAPOR
# =========================
if len(infer_times) == 0:
    print("Yeterli frame toplanamadi.")
    raise SystemExit

infer_np = np.array(infer_times)
total_np = np.array(total_times)

avg_infer = infer_np.mean()
avg_total = total_np.mean()

print("\n========== SONUC ==========")
print(f"Video              : {video_path}")
print(f"Backend            : {backend}")
print(f"Device             : {device}")
print(f"Mode               : {mode}")
print(f"det_frequency      : {det_frequency}")
print(f"Olculen frame      : {len(infer_times)}")

print(f"\nOrtalama inference suresi : {avg_infer * 1000:.2f} ms")
print(f"Inference FPS            : {1.0 / avg_infer:.2f}")

print(f"\nOrtalama end-to-end sure : {avg_total * 1000:.2f} ms")
print(f"End-to-end FPS           : {1.0 / avg_total:.2f}")

print("\n--- Inference detay ---")
print(f"Min   : {infer_np.min() * 1000:.2f} ms")
print(f"Max   : {infer_np.max() * 1000:.2f} ms")
print(f"P50   : {np.percentile(infer_np, 50) * 1000:.2f} ms")
print(f"P95   : {np.percentile(infer_np, 95) * 1000:.2f} ms")

print("\n--- End-to-end detay ---")
print(f"Min   : {total_np.min() * 1000:.2f} ms")
print(f"Max   : {total_np.max() * 1000:.2f} ms")
print(f"P50   : {np.percentile(total_np, 50) * 1000:.2f} ms")
print(f"P95   : {np.percentile(total_np, 95) * 1000:.2f} ms")