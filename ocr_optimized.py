import os
import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

# ================= 彻底关闭日志 =================
logging.getLogger("ppocr").setLevel(logging.ERROR)
for name in logging.root.manager.loggerDict:
    if "ppocr" in name:
        logging.getLogger(name).setLevel(logging.ERROR)

# ================= 配置区域 =================
INPUT_FOLDER = r"D:\pj2\images"
OUTPUT_FOLDER = r"D:\pj2\results_roi_enhance8"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# 并行 worker 数（建议设为 CPU 核心数，不超过 8）
NUM_WORKERS = min(4, os.cpu_count() or 1)

# Det 预检最少框数阈值（可根据实际数据调整）
DET_MIN_BOXES = 2

# ================= 核心函数 =================

def enhance_v1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)
    return cv2.cvtColor(sharpened.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def resize_for_ocr(img, target_size):
    h, w = img.shape[:2]
    if max(h, w) == 0:
        return img, 1.0
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w == 0 or new_h == 0:
        return img, 1.0
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4), scale


def extract_mm_number(text):
    if not text:
        return None
    matches = re.findall(r'MM[A-Z0-9]{2,}', text.upper())
    return matches[0] if matches else None


def rotate_image_fast(img, angle):
    """使用 cv2.rotate 进行直角旋转（比 warpAffine 更快），返回 (rotated_img, inverse_func)。"""
    if angle == 0:
        return img, lambda pts, orig_h, orig_w: pts
    elif angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        def inv(pts, orig_h, orig_w):
            # 顺时针 90° 逆变换：(x, y) -> (y, orig_w - 1 - x)
            return np.column_stack([pts[:, 1], orig_w - 1 - pts[:, 0]])
        return rotated, inv
    elif angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        def inv(pts, orig_h, orig_w):
            return np.column_stack([orig_w - 1 - pts[:, 0], orig_h - 1 - pts[:, 1]])
        return rotated, inv
    elif angle == 270:
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        def inv(pts, orig_h, orig_w):
            # 逆时针 90° 逆变换：(x, y) -> (orig_h - 1 - y, x)
            return np.column_stack([orig_h - 1 - pts[:, 1], pts[:, 0]])
        return rotated, inv
    else:
        raise ValueError(f"Unsupported angle: {angle}")


def save_results_async(executor, original_img, base_name, best_original_box, output_folder):
    """将图片保存任务提交到后台线程池，不阻塞主流程。"""
    def _save():
        orig_h, orig_w = original_img.shape[:2]
        result_img = original_img.copy()
        pts = best_original_box
        cv2.polylines(result_img, [pts], True, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_folder, f"result_{base_name}.jpg"), result_img)

        xs, ys = pts[:, 0], pts[:, 1]
        x1 = max(0, int(min(xs)) - 5)
        x2 = min(orig_w, int(max(xs)) + 5)
        y1 = max(0, int(min(ys)) - 5)
        y2 = min(orig_h, int(max(ys)) + 5)
        roi = original_img[y1:y2, x1:x2]
        if roi.size > 0:
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_mm.jpg"), roi)

    return executor.submit(_save)


# ================= 单张图片处理（在子进程/线程中运行）=================

def process_single_image(img_path, output_folder):
    """
    处理单张图片，返回结果字典。
    每个进程内独立初始化 OCR 实例（避免多进程共享状态问题）。
    """
    # 子进程内初始化 OCR（只初始化一次，通过全局缓存）
    ocr = _get_ocr()

    img_name = os.path.basename(img_path)
    original_img = cv2.imread(img_path)
    if original_img is None:
        return {"img_name": img_name, "found": False, "elapsed": 0.0, "error": "read failed"}

    base_name = Path(img_name).stem
    orig_h, orig_w = original_img.shape[:2]
    img_start = time.time()

    # ---- 优化点：先增强+缩放，再旋转，减少重复计算 ----
    enhanced_base = enhance_v1(original_img)
    resized_base, scale_ratio = resize_for_ocr(enhanced_base, 1920)

    best_confidence = -1.0
    best_original_box = None
    best_mm_text = ""
    found_valid = False

    angles = [0, 90, 180, 270]  # 先尝试 0°，找到即停

    for angle in angles:
        if found_valid:
            break

        # ---- 优化点：cv2.rotate 替代 warpAffine ----
        rotated_img, inv_fn = rotate_image_fast(resized_base, angle)

        # Det-Only 预检
        try:
            dt_boxes, _ = ocr.text_detector(rotated_img)
        except Exception:
            continue

        if dt_boxes is None or len(dt_boxes) < DET_MIN_BOXES:
            continue

        # 完整 OCR
        result = ocr.ocr(rotated_img)
        if not result or not result[0]:
            continue

        for line in result[0]:
            box, (txt, score) = line[0], line[1]

            # ---- 优化点：短文本提前过滤，减少正则开销 ----
            if len(txt) < 5:
                continue

            mm_number = extract_mm_number(txt.strip())
            if mm_number and len(mm_number) > 10:
                if score > best_confidence:
                    best_confidence = score
                    best_mm_text = mm_number

                    # 坐标还原：先除以缩放比，再做旋转逆变换
                    rot_box = (np.array(box, dtype=np.float32) / scale_ratio).astype(int)
                    pts_orig = inv_fn(rot_box, orig_h, orig_w).astype(int)
                    best_original_box = pts_orig

                    found_valid = True
                    if score > 0.95:
                        break

        if found_valid and best_confidence > 0.95:
            break

    img_elapsed = time.time() - img_start

    result_data = {
        "img_name": img_name,
        "base_name": base_name,
        "found": found_valid,
        "mm_text": best_mm_text,
        "confidence": best_confidence,
        "elapsed": img_elapsed,
        "box": best_original_box,
        "original_img": original_img,  # 传回用于保存（仅在同进程 I/O 场景）
    }
    return result_data


# ================= 子进程 OCR 单例缓存 =================

_ocr_instance = None

def _get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        _ocr_instance = PaddleOCR(
            use_angle_cls=True, lang="ch", ocr_version="PP-OCRv4", use_gpu=False,
            det_db_thresh=0.1, det_db_box_thresh=0.15, det_limit_side_len=2560,
            det_db_score_mode='slow', det_db_unclip_ratio=2.5, rec_batch_num=6,
            drop_score=0.3, show_log=False
        )
    return _ocr_instance


# ================= 主流程 =================

def main():
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(image_extensions)
    ]

    total_images = len(image_files)
    found_count = 0
    process_start_time = time.time()

    print("-" * 95)
    print(f"{'图片名称':<20} | {'状态':<6} | {'识别结果':<20} | {'置信度':<8} | {'耗时(s)':<8}")
    print("-" * 95)

    # ---- 优化点：异步 I/O 线程池（保存图片不阻塞主流程）----
    io_executor = ThreadPoolExecutor(max_workers=4)
    io_futures = []

    # ---- 优化点：多进程并行处理图片 ----
    # 注意：多进程时每个子进程会独立加载模型，内存消耗 = NUM_WORKERS × 单模型内存
    # 若内存紧张，可将 NUM_WORKERS 调为 1（退化为单进程，但保留其他优化）
    results_buffer = []

    worker_fn = partial(process_single_image, output_folder=OUTPUT_FOLDER)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        future_to_path = {pool.submit(worker_fn, p): p for p in image_files}
        for future in as_completed(future_to_path):
            try:
                res = future.result()
            except Exception as exc:
                img_name = os.path.basename(future_to_path[future])
                print(f"  ❌ {img_name:<16} | {'ERR':<6} | {'N/A':<20} | {'N/A':<8}     | -")
                continue

            img_name = res["img_name"]
            if res.get("error"):
                print(f"  ❌ {img_name:<16} | {'ERR':<6} | {'N/A':<20} | {'N/A':<8}     | {res['elapsed']:.3f}s")
                continue

            if res["found"]:
                print(
                    f"  ✅ {img_name:<16} | {'OK':<6} | {res['mm_text']:<20} | "
                    f"{res['confidence']:<8.4f} | {res['elapsed']:.3f}s"
                )
                found_count += 1

                # ---- 优化点：后台线程异步保存 ----
                fut = save_results_async(
                    io_executor, res["original_img"], res["base_name"],
                    res["box"], OUTPUT_FOLDER
                )
                io_futures.append(fut)
            else:
                print(
                    f"  ❌ {img_name:<16} | {'Fail':<6} | {'N/A':<20} | {'N/A':<8}     | {res['elapsed']:.3f}s"
                )

    # 等待所有图片保存完成
    for fut in io_futures:
        try:
            fut.result()
        except Exception as e:
            print(f"[警告] 保存图片时出错: {e}")

    io_executor.shutdown(wait=False)

    total_elapsed = time.time() - process_start_time
    print("-" * 95)
    print(
        f"\n📊 统计: 成功 {found_count}/{total_images}, "
        f"总耗时 {total_elapsed:.2f}s, "
        f"平均 {total_elapsed / max(1, total_images):.3f}s/张  "
        f"(并行 workers={NUM_WORKERS})"
    )
    print("完成！")


if __name__ == "__main__":
    # Windows 多进程必须在 __main__ 守护下启动
    main()
