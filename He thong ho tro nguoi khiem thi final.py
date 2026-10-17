import cv2
import threading
import pyttsx3
import asyncio
import tempfile
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog
import numpy as np
import sys
import os
import time
import csv
from datetime import datetime
from ultralytics import YOLOWorld
import torch

try:
    from locate_anything import LocateAnything

    HAS_LOCATE = True
except Exception:
    HAS_LOCATE = False

tts_lock = threading.Lock()
is_speaking = False


def speak_async(text):
    """Không giữ khóa trong lúc phát âm, tránh chặn luồng camera."""
    global is_speaking
    if not text:
        return False
    with tts_lock:
        if is_speaking:
            return False
        is_speaking = True

    def _run():
        global is_speaking
        audio_path = None
        try:
            try:
                import edge_tts
                import pygame

                async def make_audio():
                    nonlocal audio_path
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp:
                        audio_path = temp.name
                    await edge_tts.Communicate(text, "vi-VN-HoaiMyNeural").save(audio_path)

                asyncio.run(make_audio())
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                pygame.mixer.music.unload()
            except Exception as edge_error:
                print(f"[TTS] Edge TTS không hoạt động, thử giọng offline: {edge_error}")
                engine = pyttsx3.init()
                for voice in engine.getProperty('voices'):
                    desc = (voice.id + " " + voice.name).lower()
                    if "vietnam" in desc or "vi-vn" in desc:
                        engine.setProperty('voice', voice.id)
                        break
                engine.setProperty('rate', 155)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except Exception as exc:
            print(f"[TTS] Không thể phát cảnh báo: {exc}")
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            with tts_lock:
                is_speaking = False

    threading.Thread(target=_run, daemon=True).start()
    return True


def put_text_vi(image, text, org, fontFace=None, fontScale=0.5, color=(255, 255, 255), thickness=1, **kwargs):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font_size = max(12, int(30 * fontScale))
        font_path = r"C:\Windows\Fonts\arial.ttf"
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        x, baseline_y = org
        y = max(0, baseline_y - font_size)
        rgb_color = (int(color[2]), int(color[1]), int(color[0]))
        for offset in range(max(1, int(thickness))):
            draw.text((x, y), str(text), font=font, fill=rgb_color, stroke_width=offset)
        image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        pass


current_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(current_dir, "nhat_ky_thu_nghiem.csv")

LOG_HEADERS = [
    "ThoiGian", "TinhHuong", "NguonHinhAnh", "ChatLuongAnh",
    "LanTrai", "LanGiua", "LanPhai", "CuLyGiua", "CanhBaoAmThanh", "DanhGiaThucTe"
]


def init_csv_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADERS)


def record_test_log(scenario, source_name, quality_desc, left_st, center_st, right_st, c_zone, alert_msg, eval_text):
    init_csv_log()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        now_str, scenario, source_name, quality_desc,
        left_st, center_st, right_st, c_zone if c_zone else "Thông thoáng",
        alert_msg, eval_text
    ]
    try:
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"Lỗi ghi CSV: {e}")
        return False


def count_log_stats():
    if not os.path.exists(LOG_FILE):
        return {"total": 0, "dung": 0, "nham": 0, "sot": 0}
    total, dung, nham, sot = 0, 0, 0, 0
    try:
        with open(LOG_FILE, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                total += 1
                ev = r.get("DanhGiaThucTe", "")
                if "Đúng" in ev:
                    dung += 1
                elif "Báo nhầm" in ev:
                    nham += 1
                elif "Bỏ sót" in ev:
                    sot += 1
    except Exception:
        pass
    return {"total": total, "dung": dung, "nham": nham, "sot": sot}


model_path = os.path.join(current_dir, 'yolov8s-world.pt')
if not os.path.exists(model_path):
    model_path = 'yolov8s-world.pt'

print("Đang khởi tạo mô hình YOLOWorld...")
model = YOLOWorld(model_path)

CUSTOM_CLASSES = [
    "cabinet", "electric box", "obstacle", "barrier", "fence",
    "box", "barrel", "sign", "person", "vehicle", "wall", "pillar",
    "chair", "table", "desk", "door", "stairs", "steps", "backpack", "trash can",
    "utility pole", "electric pole", "streetlight pole"
]
model.set_classes(CUSTOM_CLASSES)

VIETNAMESE_NAMES = {
    "cabinet": "tủ", "electric box": "hộp điện", "obstacle": "vật cản",
    "barrier": "rào chắn", "fence": "hàng rào", "box": "thùng hàng",
    "barrel": "thùng phi", "sign": "biển báo", "person": "người",
    "vehicle": "xe", "wall": "bức tường", "pillar": "cột",
    "chair": "ghế", "table": "bàn", "desk": "bàn học",
    "door": "cửa", "stairs": "cầu thang", "steps": "bậc thang",
    "backpack": "ba lô", "trash can": "thùng rác",
    "utility pole": "cột điện", "electric pole": "cột điện", "streetlight pole": "cột đèn"
}

# ĐÃ NÂNG NGƯỠNG TÍN CẬY ĐỂ HẠN CHẾ BÁO GIẢ
CONF_CONFIRMED = 0.35
CONF_UNCERTAIN = 0.18
INFER_SIZE = 384  # Tang len 416/512 neu bo sot cot nho
DETECT_EVERY = 3  # Khong tang qua cao khi dung cho canh bao vat can
MAX_TRACK_AGE = 0.7  # giay, khong tin vao bounding box cu qua lau

SCENARIOS = ["Hành lang", "Lớp học", "Cầu thang"]


def check_image_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_dark = brightness < 40.0
    is_overexposed = brightness > 230.0
    is_blurry = blur_score < 60.0
    is_degraded = is_dark or is_overexposed or is_blurry

    if is_dark:
        desc = f"Quá tối ({brightness:.1f})"
    elif is_overexposed:
        desc = f"Quá chói ({brightness:.1f})"
    elif is_blurry:
        desc = f"Mờ rung ({blur_score:.1f})"
    else:
        desc = "Rõ nét"

    return is_degraded, desc, is_dark, is_blurry


def get_proximity_zone(y2, box_h, h_frame, danger_line_y):
    # Chỉ là ước lượng tương đối theo vị trí đáy hộp, KHÔNG phải khoảng cách mét.
    # Tránh yêu cầu vật thể phải cao >=30% ảnh: sẽ bỏ sót vật thấp sát chân.
    foot_ratio = max(0.0, min(1.0, y2 / max(1, h_frame)))
    if foot_ratio >= 0.82:
        return "NEAR"
    if foot_ratio >= 0.58:
        return "AHEAD"
    return "FAR"


def open_camera_stream(src):
    print(f"Đang kết nối nguồn camera trực tiếp: {src} ...")
    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap, src
        cap.release()

    if src != 0:
        print("Tự động chuyển sang Webcam máy tính (0)...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, 0
            cap.release()

    return None, None


class SparseFlowTracker:
    """ByteTrack gan ID tai frame YOLO; optical flow dich chuyen box giua cac frame."""
    def __init__(self):
        self.gray = None
        self.objects = []
        self.last_detect = 0.0

    def reset(self):
        self.gray = None
        self.objects = []
        self.last_detect = 0.0

    def update(self, frame, run_detector, device):
        # Optical flow o do phan giai thap de giam CPU
        flow_scale = 0.5
        small = cv2.resize(frame, None, fx=flow_scale, fy=flow_scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        now = time.monotonic()
        if run_detector:
            # Ultralytics tich hop ByteTrack, dung tracker config co san.
            result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 conf=CONF_UNCERTAIN, iou=0.40,
                                 imgsz=INFER_SIZE, device=device, verbose=False)[0]
            self.objects = []
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
                for xyxy, conf, cls, tid in zip(boxes.xyxy.cpu().numpy(),
                                                boxes.conf.cpu().numpy(),
                                                boxes.cls.int().cpu().numpy(), ids):
                    self.objects.append((xyxy.astype(float), float(conf), int(cls), int(tid)))
            self.last_detect = now
        elif self.gray is not None and self.objects and now - self.last_detect <= MAX_TRACK_AGE:
            # Optical flow la uoc luong, KHONG phai phat hien vat can moi.
            prev_pts = []
            for box, _, _, _ in self.objects:
                x1, y1, x2, y2 = box
                prev_pts.extend([[(x1+x2)/2, (y1+y2)/2], [x1,y1], [x2,y2]])
            prev_pts = np.array(prev_pts, dtype=np.float32).reshape(-1,1,2) * flow_scale
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, prev_pts, None,
                                                       winSize=(17,17), maxLevel=2)
            if nxt is not None and status is not None:
                updated = []
                for i, (box, conf, cls, tid) in enumerate(self.objects):
                    valid = status[3*i:3*i+3].reshape(-1).astype(bool)
                    if valid.any():
                        shift = np.median((nxt[3*i:3*i+3] - prev_pts[3*i:3*i+3]).reshape(-1,2)[valid], axis=0)
                        shift = np.clip(shift / flow_scale, -30, 30)
                        moved = box + np.array([shift[0],shift[1],shift[0],shift[1]])
                        moved[[0,2]] = np.clip(moved[[0,2]], 0, frame.shape[1]-1)
                        moved[[1,3]] = np.clip(moved[[1,3]], 0, frame.shape[0]-1)
                        updated.append((moved, conf, cls, tid))
                self.objects = updated
        else:
            self.objects = []
        self.gray = gray
        return self.objects


def process_frame_lanes(frame, detections=None, device='cpu'):
    h, w = frame.shape[:2]
    one_third = w // 3
    two_thirds = 2 * (w // 3)
    danger_y = int(h * 0.46)

    is_degraded, quality_desc, is_dark, is_blurry = check_image_quality(frame)

    # Ket qua ByteTrack/optical flow duoc chuyen vao tu main, khong chay YOLO lap lai.
    detections = detections or []

    left_blocked, center_blocked, right_blocked = False, False, False
    left_caution, center_caution, right_caution = False, False, False
    closest_center_zone = None
    closest_left_zone = None
    closest_right_zone = None

    detected_stairs = False
    detected_objects_summary = []
    pole_classes = {"utility pole", "electric pole", "streetlight pole", "pillar"}

    if detections:
        boxes = [np.asarray(d[0]).astype(int) for d in detections]
        confs = [d[1] for d in detections]
        clss = [d[2] for d in detections]
        track_ids = [d[3] for d in detections]

        for box, conf, cls_id, tid in zip(boxes, confs, clss, track_ids):
            x1, y1, x2, y2 = box
            box_w = max(x2 - x1, 1)
            box_h = max(y2 - y1, 1)
            area = box_w * box_h

            cls_name = CUSTOM_CLASSES[cls_id] if 0 <= cls_id < len(CUSTOM_CLASSES) else "obstacle"
            is_pole = cls_name in pole_classes
            # Cột điện mảnh không được loại chỉ vì diện tích bounding box nhỏ.
            min_area_ratio = 0.0003 if is_pole else 0.002
            if y2 < int(h * 0.25) or area < (w * h * min_area_ratio) or conf < CONF_UNCERTAIN:
                continue
            is_confirmed = (conf >= (0.28 if is_pole else CONF_CONFIRMED))
            if cls_name in ["stairs", "steps"] and is_confirmed and y2 >= int(h * 0.40):
                detected_stairs = True

            zone = get_proximity_zone(y2, box_h, h, danger_y)
            cx = (x1 + x2) // 2

            overlap_left = max(x1, one_third)
            overlap_right = min(x2, two_thirds)
            mid_overlap = max(0, overlap_right - overlap_left)
            overlap_ratio = mid_overlap / box_w

            in_center = (overlap_ratio >= (0.05 if is_pole else 0.15)) or (one_third <= cx <= two_thirds)
            is_danger = (y2 >= danger_y) or (in_center and box_h > int(h * 0.25))

            left_overlap = max(0, min(x2, one_third) - max(x1, 0))
            right_overlap = max(0, min(x2, w) - max(x1, two_thirds))
            in_left = (left_overlap / box_w >= 0.20) or (cx < one_third)
            in_right = (right_overlap / box_w >= 0.20) or (cx > two_thirds)

            viet_name = VIETNAMESE_NAMES.get(cls_name, cls_name)
            lane_str = "ở giữa" if in_center else ("bên trái" if in_left else "bên phải")
            detected_objects_summary.append(f"{viet_name} ({lane_str}, {zone})")
            if is_pole:
                pass  # Bo log tung frame de giam overhead

            if is_danger:
                if in_center:
                    if is_confirmed:
                        center_blocked = True
                        if closest_center_zone is None or {"FAR": 0, "AHEAD": 1, "NEAR": 2}[zone] > \
                                {"FAR": 0, "AHEAD": 1, "NEAR": 2}[closest_center_zone]:
                            closest_center_zone = zone
                    else:
                        center_caution = True
                        if not closest_center_zone: closest_center_zone = zone
                if in_left:
                    if is_confirmed:
                        left_blocked = True
                        if closest_left_zone != "NEAR": closest_left_zone = zone
                    else:
                        left_caution = True
                if in_right:
                    if is_confirmed:
                        right_blocked = True
                        if closest_right_zone != "NEAR": closest_right_zone = zone
                    else:
                        right_caution = True

                box_color = (0, 0, 255) if is_confirmed else (0, 165, 255)
                label_text = f"{viet_name.upper()} [{zone}] ({int(conf * 100)}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                # Nhan ngan bang OpenCV: tranh BGR->PIL->BGR moi vat the
                fast_label = f"{cls_name.upper()} {int(conf * 100)}% ID:{tid}" if tid >= 0 else f"{cls_name.upper()} {int(conf * 100)}%"
                cv2.putText(frame, fast_label, (x1, max(y1 - 7, 48)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)

    speech_text = ""
    status_text = ""
    status_color = (0, 255, 0)

    if is_degraded:
        status_color = (0, 165, 255)
        status_text = f"KHÔNG NHÌN RÕ ({quality_desc.upper()}) - LUÔN DÙNG GẬY"
        speech_text = "Không nhìn rõ, hãy cẩn thận. Luôn dùng gậy dò đường."
    elif detected_stairs:
        speech_text = "Chú ý! Có cầu thang phía trước, hãy dò gậy cẩn thận."
        status_text = "CẢNH BÁO: CẦU THANG PHÍA TRƯỚC"
        status_color = (0, 0, 255)
    elif center_blocked:
        status_color = (0, 0, 255)
        dist_vi = "rất gần" if closest_center_zone == "NEAR" else (
            "trước mặt" if closest_center_zone == "AHEAD" else "ở xa")
        if not left_blocked and right_blocked:
            status_text = f"LÀN GIỮA BỊ CHẶN [{closest_center_zone}] | TRÁI THOÁNG"
            speech_text = f"Cẩn thận, có vật cản {dist_vi} trên đường đi. Không tự rẽ nếu chưa kiểm tra lối đi."
        elif not right_blocked and left_blocked:
            status_text = f"LÀN GIỮA BỊ CHẶN [{closest_center_zone}] | PHẢI THOÁNG"
            speech_text = f"Cẩn thận, có vật cản {dist_vi} trên đường đi. Không tự rẽ nếu chưa kiểm tra lối đi."
        elif not left_blocked and not right_blocked:
            status_text = f"LÀN GIỮA BỊ CHẶN [{closest_center_zone}] | HAI BÊN THOÁNG"
            speech_text = f"Cẩn thận, có vật cản {dist_vi} trên đường đi. Hai bên chưa phát hiện vật cản, cần kiểm tra bằng gậy."
        else:
            status_text = "TOÀN BỘ LỐI ĐI BỊ CHẶN | DỪNG LẠI"
            speech_text = "Cảnh báo, đường đi bị chặn hoàn toàn. Hãy dừng lại."
    elif center_caution:
        status_color = (0, 165, 255)
        status_text = "CHÚ Ý: NGHI NGỜ CÓ VẬT CẢN TRÊN ĐƯỜNG"
        speech_text = "Chú ý, có thể có vật cản trên đường đi. Đi chậm và dùng gậy."
    elif left_blocked or left_caution or right_blocked or right_caution:
        status_color = (0, 220, 220)
        if (left_blocked or left_caution) and (right_blocked or right_caution):
            status_text = "ĐƯỜNG GIỮA THOÁNG | VẬT CẢN HAI BÊN"
            speech_text = "Đường giữa thông thoáng. Có vật cản ở cả hai bên."
        elif left_blocked or left_caution:
            status_text = "ĐƯỜNG GIỮA THOÁNG | CÓ VẬT CẢN BÊN TRÁI"
            speech_text = "Đường giữa thông thoáng. Có vật cản bên trái."
        else:
            status_text = "ĐƯỜNG GIỮA THOÁNG | CÓ VẬT CẢN BÊN PHẢI"
            speech_text = "Đường giữa thông thoáng. Có vật cản bên phải."
    else:
        status_color = (0, 255, 0)
        status_text = "CHƯA PHÁT HIỆN VẬT CẢN"
        speech_text = "Camera chưa phát hiện vật cản, hãy tiếp tục dùng gậy kiểm tra."

    return {
        "frame": frame,
        "is_degraded": is_degraded,
        "quality_desc": quality_desc,
        "left_blocked": left_blocked, "left_caution": left_caution, "left_zone": closest_left_zone,
        "center_blocked": center_blocked, "center_caution": center_caution, "center_zone": closest_center_zone,
        "right_blocked": right_blocked, "right_caution": right_caution, "right_zone": closest_right_zone,
        "status_text": status_text,
        "status_color": status_color,
        "speech_text": speech_text,
        "danger_y": danger_y,
        "one_third": one_third,
        "two_thirds": two_thirds,
        "objects_summary": detected_objects_summary
    }


def render_hud(analysis, scenario, source_label, stats):
    """HUD dễ đọc: một lần chuyển PIL, màu nhất quán, không che vùng quan sát chính."""
    frame = analysis["frame"]
    h, w = frame.shape[:2]
    if h < 200 or w < 300:
        return
    center = analysis["center_blocked"]
    caution = analysis["center_caution"] or analysis["is_degraded"]
    stairs = "CẦU THANG" in analysis["status_text"].upper()
    if center or stairs:
        headline, accent = "CẢNH BÁO VẬT CẢN", (235, 67, 72)
        if stairs:
            headline = "CHÚ Ý CẦU THANG"
    elif caution:
        headline, accent = "CẦN KIỂM TRA", (246, 169, 53)
    else:
        headline, accent = "ĐANG QUAN SÁT", (65, 198, 151)

    # Vẽ vùng chia làn mờ để không che vật cản.
    for x in (analysis["one_third"], analysis["two_thirds"]):
        cv2.line(frame, (x, 62), (x, h - 112), (115, 125, 135), 1)
    # Chuyển đổi màu và vẽ tất cả chữ trong một lượt (nhanh hơn nhiều lượt put_text_vi).
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = max(0.72, min(1.15, w / 960))
    def font(sz, bold=False):
        key = (max(11, int(sz * scale)), bold)
        if key not in _HUD_FONTS:
            candidates = ([r"C:\Windows\Fonts\arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
                          if bold else [r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
            for path in candidates:
                try:
                    _HUD_FONTS[key] = ImageFont.truetype(path, key[0]); break
                except OSError:
                    pass
            else:
                _HUD_FONTS[key] = ImageFont.load_default()
        return _HUD_FONTS[key]
    def txt(x, y, value, size=19, fill=(255,255,255), bold=False, max_width=None):
        value = str(value)
        f = font(size, bold)
        if max_width:
            while value and draw.textbbox((0,0), value, font=f)[2] > max_width:
                value = value[:-2].rstrip() + "…"
        draw.text((int(x), int(y)), value, font=f, fill=fill)
    pad = max(12, int(w * .018))
    header_h = max(56, int(65 * scale))
    draw.rounded_rectangle((pad, 8, w-pad, header_h), radius=14, fill=(15, 24, 36, 223))
    draw.rounded_rectangle((pad+10, 18, pad+17, header_h-10), radius=3, fill=accent+(255,))
    txt(pad+29, 16, headline, 24, accent, True, w * .67)
    txt(w-pad-175*scale, 24, "AI VISION  •  LIVE", 12, (195,207,219))

    lane_top = header_h + 8
    lane_h = max(43, int(51*scale))
    lanes = [("TRÁI", "left"), ("GIỮA", "center"), ("PHẢI", "right")]
    gap = 7
    lane_w = (w-2*pad-2*gap)/3
    for i,(label,k) in enumerate(lanes):
        x = pad+i*(lane_w+gap)
        blocked = analysis[k+"_blocked"]
        uncertain = analysis[k+"_caution"]
        status = "VẬT CẢN" if blocked else ("CHÚ Ý" if uncertain else "CHƯA PHÁT HIỆN")
        color = (235,67,72) if blocked else ((246,169,53) if uncertain else (65,198,151))
        draw.rounded_rectangle((x,lane_top,x+lane_w,lane_top+lane_h),radius=10,fill=(16,26,39,218))
        draw.ellipse((x+9,lane_top+12,x+18,lane_top+21),fill=color+(255,))
        txt(x+24,lane_top+7,label,14,(232,239,248),True)
        txt(x+24,lane_top+25,status,12,color,False,lane_w-31)

    # Thẻ cảnh báo ở dưới, không che vật thể ở giữa ảnh.
    footer_h = max(99, int(116*scale))
    fy = h-footer_h-8
    draw.rounded_rectangle((pad,fy,w-pad,h-8),radius=15,fill=(12,20,32,230))
    draw.rounded_rectangle((pad+10,fy+12,pad+17,h-20),radius=3,fill=accent+(255,))
    zone = analysis.get("center_zone")
    zone_vi = {"NEAR":"ƯỚC LƯỢNG: GẦN", "AHEAD":"ƯỚC LƯỢNG: PHÍA TRƯỚC", "FAR":"ƯỚC LƯỢNG: XA"}.get(zone, "CHƯA XÁC ĐỊNH CỰ LY")
    txt(pad+29,fy+10,zone_vi,15,accent,True,w-2*pad-44)
    # Không khẳng định lối đi an toàn chỉ từ việc không có bounding box.
    message = ("DỪNG LẠI • DÙNG GẬY KIỂM TRA" if center or stairs else
               "CAMERA NHÌN KHÔNG RÕ • KIỂM TRA" if analysis["is_degraded"] else
               "CÓ THỂ CÓ VẬT CẢN • KIỂM TRA" if caution else
               "CHƯA PHÁT HIỆN VẬT CẢN PHÍA TRƯỚC")
    txt(pad+29,fy+34,message,19,(248,250,252),True,w-2*pad-43)
    txt(pad+29,fy+62,f"{scenario}  |  {source_label}  |  Ảnh: {analysis['quality_desc']}",12,(186,200,215),False,w-2*pad-43)
    txt(pad+29,fy+83,"1 Đúng  2 Nhầm  3 Sót  T Log  C Chế độ  S Đọc  V Video  P Ảnh  Q Thoát",11,(169,186,201),False,w-2*pad-43)
    im = Image.alpha_composite(im,layer).convert("RGB")
    frame[:] = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


_HUD_FONTS = {}

def main():
    print("==================================================")
    print("  PHẦN MỀM HỖ TRỢ NGƯỜI KHIẾM THỊ ")
    print("==================================================")
    init_csv_log()

    initial_prompt = "Hệ thống hỗ trợ di chuyển đã bật. Lưu ý quan trọng: Luôn sử dụng kèm gậy dẫn đường, AI chỉ mang tính hỗ trợ."
    print(f"\n🔊 {initial_prompt}\n")
    speak_async(initial_prompt)

    camera_src = "http://192.168.1.42:8080/video"
    cap, actual_src = open_camera_stream(camera_src)

    scenario_idx = 0
    current_scenario = SCENARIOS[scenario_idx]
    stats = count_log_stats()

    last_speech_time = 0.0
    speech_cooldown = 3.0
    previous_alert = ""
    stable_alert_frames = 0
    last_announced_alert = ""
    last_announce_at = 0.0
    last_alert_level = 0
    clear_frames = 0
    fps_ema = 0.0
    tracker = SparseFlowTracker()
    frame_index = 0
    previous_source = actual_src

    WINDOW_NAME = "Hỗ trợ di chuyển - Bản hoàn chỉnh 2026"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    ai_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Mô hình đang chạy trên thiết bị: {ai_device.upper()}")

    print("=" * 60)
    print("HƯỚNG DẪN ĐIỀU KHIỂN & KIỂM THỬ:")
    print(" - Phím 1 : Đánh giá ĐÚNG")
    print(" - Phím 2 : Đánh giá BÁO NHẦM")
    print(" - Phím 3 : Đánh giá BỎ SÓT")
    print(" - Phím T : Ghi nhanh nhật ký thử nghiệm ra file CSV")
    print(" - Phím C : Đổi tình huống (Hành lang -> Lớp học -> Cầu thang)")
    print(" - Phím S : Đọc tóm tắt các vật cản đang có trước mặt")
    print(" - Phím V : Mở file video (.mp4/.avi) để test kiểm thử mô hình")
    print(" - Phím P : Mở ảnh tĩnh để kiểm tra")
    print(" - Phím Q : Thoát chương trình")
    print("=" * 60 + "\n")

    current_frame = None

    while True:
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                if isinstance(actual_src, str) and not actual_src.startswith("http"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Mất tín hiệu camera trực tiếp, đang kết nối lại...")
                time.sleep(1.0)
                cap.release()
                cap, actual_src = open_camera_stream(camera_src)
                continue
            current_frame = frame
        else:
            if current_frame is None:
                break

        h_orig, w_orig = current_frame.shape[:2]
        # Không phóng ảnh nhỏ lên 960px: tốn RAM và thời gian xử lý.
        # Giới hạn ảnh hiển thị, model tự resize xuống INFER_SIZE.
        if w_orig > 960:
            scale = 960 / w_orig
            current_frame = cv2.resize(current_frame, (960, int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        tick_start = time.perf_counter()

        # YOLO + ByteTrack moi DETECT_EVERY frame; optical flow tren frame xen ke.
        if actual_src != previous_source:
            tracker.reset()
            # Xoa trang thai ByteTrack khi doi nguon video.
            if getattr(model, "predictor", None) is not None and hasattr(model.predictor, "trackers"):
                for bt in model.predictor.trackers:
                    bt.reset()
            frame_index = 0
            previous_source = actual_src
        is_still = isinstance(actual_src, str) and actual_src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        run_detector = (frame_index == 0 if is_still else frame_index % DETECT_EVERY == 0)
        detections = tracker.update(current_frame, run_detector, ai_device)
        frame_index += 1
        analysis = process_frame_lanes(current_frame.copy(), detections=detections, device=ai_device)

        now = time.time()
        # Chỉ đếm những lần YOLO thực sự phát hiện.
        # Các frame Optical Flow KHÔNG được đặt bộ đếm về 0.
        alert_key = (
            "visibility" if analysis["is_degraded"] else
            "stairs" if "cầu thang" in analysis["speech_text"].lower() else
            "center" if analysis["center_blocked"] else
            "uncertain" if analysis["center_caution"] else
            "side" if (analysis["left_blocked"] or analysis["right_blocked"] or
                       analysis["left_caution"] or analysis["right_caution"]) else
            "clear"
        )
        if run_detector:
            if alert_key == previous_alert:
                stable_alert_frames += 1
            else:
                previous_alert = alert_key
                stable_alert_frames = 1

        # Chỉ phát âm khi nguy cơ mới xuất hiện hoặc mức nguy hiểm tăng.
        # Không lặp lại cảnh báo cũ theo chu kỳ 12 giây.
        if analysis["is_degraded"]:
            alert_level, alert_kind = 2, "visibility"
            spoken = "Camera nhìn không rõ. Hãy dừng lại và dùng gậy kiểm tra."
        elif "cầu thang" in analysis["speech_text"].lower():
            alert_level, alert_kind = 3, "stairs"
            spoken = "Chú ý có cầu thang. Hãy dừng lại và kiểm tra bằng gậy."
        elif analysis["center_blocked"]:
            z = analysis["center_zone"]
            alert_level = 3 if z == "NEAR" else 2 if z == "AHEAD" else 1
            alert_kind = "center_obstacle"
            side_note = ""
            # Chỉ mô tả quan sát, không chỉ dẫn rẽ vào nơi camera chưa xác minh.
            if not analysis["left_blocked"] and not analysis["left_caution"]:
                side_note += " Camera chưa phát hiện vật cản bên trái."
            if not analysis["right_blocked"] and not analysis["right_caution"]:
                side_note += " Camera chưa phát hiện vật cản bên phải."
            spoken = ("Có vật cản phía trước" +
                      (", ở rất gần." if alert_level == 3 else ".") +
                      " Hãy dừng lại, dùng gậy kiểm tra." + side_note +
                      " Không tự rẽ khi chưa xác định được lối đi an toàn.")
        elif analysis["center_caution"]:
            alert_level, alert_kind = 1, "uncertain_obstacle"
            spoken = "Có thể có vật cản phía trước. Hãy dùng gậy kiểm tra."
        elif analysis["left_blocked"] or analysis["right_blocked"]:
            alert_level, alert_kind = 1, "side_obstacle"
            side = ("hai bên" if analysis["left_blocked"] and analysis["right_blocked"]
                    else "bên trái" if analysis["left_blocked"] else "bên phải")
            spoken = f"Chú ý, camera phát hiện vật cản {side}. Hãy dùng gậy kiểm tra."
        else:
            alert_level, alert_kind, spoken = 0, "clear", ""

        if alert_level == 0:
            if run_detector:
                clear_frames += 1
            # Chỉ đặt lại trạng thái sau một khoảng đủ dài không thấy nguy cơ.
            if clear_frames >= 12:
                last_announced_alert = ""
                last_alert_level = 0
        else:
            clear_frames = 0
            changed = alert_kind != last_announced_alert
            escalated = (alert_kind == last_announced_alert and
                         alert_level > last_alert_level)
            confirmations_needed = 1 if is_still or alert_kind == "stairs" else 2
            if (stable_alert_frames >= confirmations_needed and (changed or escalated)
                    and now - last_speech_time >= speech_cooldown):
                # Chỉ ghi nhận đã thông báo khi giọng nói thực sự được nhận.
                if speak_async(spoken):
                    print(f"[CẢNH BÁO] {spoken}")
                    last_speech_time = now
                    last_announce_at = now
                    last_announced_alert = alert_kind
                    last_alert_level = alert_level

        source_label = "Webcam/IP" if actual_src in [0, 1] or str(actual_src).startswith("http") else os.path.basename(
            str(actual_src))
        render_hud(analysis, current_scenario, source_label, stats)
        # Vẽ HUD trước khi thu nhỏ/hiển thị, tránh dùng khung hình chưa có HUD.
        display_frame = analysis["frame"]
        h_disp, w_disp = display_frame.shape[:2]
        if w_disp > 1280:
            scale_percent = 1280 / w_disp
            display_frame = cv2.resize(display_frame, (1280, int(h_disp * scale_percent)), interpolation=cv2.INTER_AREA)
        elapsed = max(time.perf_counter() - tick_start, 1e-6)
        fps_now = 1.0 / elapsed
        fps_ema = fps_now if fps_ema == 0 else (0.85 * fps_ema + 0.15 * fps_now)
        cv2.putText(display_frame, f"FPS: {fps_ema:.1f} | Xu ly: {elapsed * 1000:.0f} ms",
                    (12, max(58, display_frame.shape[0] - 54)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('t'):
            record_test_log(
                current_scenario, source_label, analysis["quality_desc"],
                "Chặn" if analysis["left_blocked"] else "Thoáng",
                "Chặn" if analysis["center_blocked"] else "Thoáng",
                "Chặn" if analysis["right_blocked"] else "Thoáng",
                analysis["center_zone"], analysis["speech_text"], "Chưa đánh giá"
            )
            stats = count_log_stats()
            print("📝 Đã ghi nhật ký thành công vào nhat_ky_thu_nghiem.csv")
            speak_async("Đã ghi nhật ký.")
        elif key == ord('1'):
            record_test_log(
                current_scenario, source_label, analysis["quality_desc"],
                "Chặn" if analysis["left_blocked"] else "Thoáng",
                "Chặn" if analysis["center_blocked"] else "Thoáng",
                "Chặn" if analysis["right_blocked"] else "Thoáng",
                analysis["center_zone"], analysis["speech_text"], "Đúng"
            )
            stats = count_log_stats()
            print("✅ Đã ghi nhận: ĐÚNG")
            speak_async("Ghi nhận đúng.")
        elif key == ord('2'):
            record_test_log(
                current_scenario, source_label, analysis["quality_desc"],
                "Chặn" if analysis["left_blocked"] else "Thoáng",
                "Chặn" if analysis["center_blocked"] else "Thoáng",
                "Chặn" if analysis["right_blocked"] else "Thoáng",
                analysis["center_zone"], analysis["speech_text"], "Báo nhầm"
            )
            stats = count_log_stats()
            print("⚠️ Đã ghi nhận: BÁO NHẦM")
            speak_async("Ghi nhận báo nhầm.")
        elif key == ord('3'):
            record_test_log(
                current_scenario, source_label, analysis["quality_desc"],
                "Chặn" if analysis["left_blocked"] else "Thoáng",
                "Chặn" if analysis["center_blocked"] else "Thoáng",
                "Chặn" if analysis["right_blocked"] else "Thoáng",
                analysis["center_zone"], analysis["speech_text"], "Bỏ sót"
            )
            stats = count_log_stats()
            print("🚨 Đã ghi nhận: BỎ SÓT")
            speak_async("Ghi nhận bỏ sót.")
        elif key == ord('c'):
            scenario_idx = (scenario_idx + 1) % len(SCENARIOS)
            current_scenario = SCENARIOS[scenario_idx]
            msg = f"Chuyển sang tình huống {current_scenario}"
            print(f"🔄 {msg}")
            speak_async(msg)
        elif key == ord('s'):
            if analysis["objects_summary"]:
                msg = "Phát hiện: " + ", ".join(analysis["objects_summary"][:4])
            else:
                msg = "Camera chưa phát hiện vật cản. Hãy tiếp tục dùng gậy kiểm tra."
            print(f"🔊 {msg}")
            speak_async(msg)
        elif key == ord('v'):
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            vid_p = filedialog.askopenfilename(
                title="Chọn file video kiểm tra",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")]
            )
            root.destroy()
            if vid_p and os.path.exists(vid_p):
                if cap: cap.release()
                cap = cv2.VideoCapture(vid_p)
                actual_src = vid_p
                print(f"Đã nạp video kiểm tra: {os.path.basename(vid_p)}")
                speak_async("Đã nạp video kiểm tra.")
        elif key == ord('p'):
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            img_p = filedialog.askopenfilename(
                title="Chọn ảnh kiểm tra",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")]
            )
            root.destroy()
            if img_p and os.path.exists(img_p):
                loaded_img = cv2.imread(img_p)
                if loaded_img is not None:
                    if cap:
                        cap.release()
                    cap = None
                    current_frame = loaded_img
                    actual_src = img_p
                    print(f"Đã nạp ảnh kiểm tra: {os.path.basename(img_p)}")
                    speak_async("Đã nạp ảnh kiểm tra.")

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

