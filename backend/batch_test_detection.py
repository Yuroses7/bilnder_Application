"""
batch_test_detection.py
------------------------
ทดสอบ API detection กับรูปทั้งหมดในโฟลเดอร์ที่กำหนด
- ส่งรูปไปที่ /detectpostman
- poll /status จนเสร็จ
- ดึงผลจาก /result และ /analysis
- เก็บเวลา: yolo, depth, ai_scene (gemini), speech, total
- แสดงรูปพร้อม bounding box + สรุปผล (matplotlib)
- เซฟสรุปทั้งหมดเป็น CSV

วิธีใช้:
    python batch_test_detection.py --folder ./test_images --base-url http://localhost:8000

ต้องติดตั้ง:
    pip install requests matplotlib pillow pandas tqdm --break-system-packages
"""

import argparse
import csv
import time
from pathlib import Path

import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# ตั้งค่า font ให้รองรับภาษาไทย (ไม่งั้น matplotlib จะโชว์เป็นสัญลักษณ์เพี้ยน/กล่อง)
# หาตามลำดับ: font ไทยที่มักมีอยู่แล้วใน Windows -> ถ้าไม่เจอ fallback เป็น default
# ---------------------------------------------------------------------------
def setup_thai_font():
    thai_font_candidates = [
        "Tahoma", "Leelawadee UI", "Leelawadee", "Angsana New",
        "Cordia New", "TH Sarabun New", "Noto Sans Thai",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in thai_font_candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            print(f"✓ ใช้ font: {name} (รองรับภาษาไทย)")
            return
    print("⚠️ ไม่พบ font ไทยในเครื่อง ตัวอักษรไทยในรูปอาจแสดงผลไม่ถูกต้อง "
          "(แนะนำติดตั้ง font เช่น 'TH Sarabun New' หรือ 'Noto Sans Thai')")


setup_thai_font()


def poll_status(base_url: str, job_id: str, timeout: float = 60.0, interval: float = 0.5) -> dict:
    """รอจนกว่า job จะ completed หรือ failed หรือ timeout"""
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{base_url}/status/{job_id}")
        r.raise_for_status()
        status = r.json()
        if status["status"] in ("completed", "failed"):
            return status
        time.sleep(interval)
    raise TimeoutError(f"job {job_id} ไม่เสร็จภายใน {timeout} วิ")


def test_one_image(base_url: str, image_path: Path, enable_speech: bool = True) -> dict:
    """ส่งรูปหนึ่งรูปไปเทส แล้วคืนค่าผลลัพธ์รวม timing"""
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        params = {"enable_speech": str(enable_speech).lower()}
        r = requests.post(f"{base_url}/detectpostman", files=files, params=params)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    status = poll_status(base_url, job_id)

    if status["status"] == "failed":
        return {
            "image": image_path.name,
            "job_id": job_id,
            "status": "failed",
            "error": status.get("error", ""),
            "detections": [],
            "timing": {},
            "speech_text": None,
        }

    result_r = requests.get(f"{base_url}/result/{job_id}")
    result_r.raise_for_status()
    result = result_r.json()

    analysis_r = requests.get(f"{base_url}/analysis/{job_id}")
    analysis = analysis_r.json() if analysis_r.status_code == 200 else {}

    return {
        "image": image_path.name,
        "job_id": job_id,
        "status": "completed",
        "detections": result.get("results", []),
        "timing": result.get("timing", {}),
        "speech_text": result.get("speech_text"),
        "scene_analysis": analysis.get("scene_analysis", {}),
        "base_url": base_url,
    }


def show_result(image_path: Path, result: dict):
    """แสดงรูป พร้อม bounding box และสรุปเวลาแต่ละขั้นตอน"""
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(9, 7))
    ax.imshow(img)

    for det in result["detections"]:
        x1, y1, x2, y2 = det["box"]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        label_text = f'{det["label"]} {det["distance"]}m ({det["confidence"]:.2f})'
        ax.text(
            x1, max(0, y1 - 5), label_text,
            color="black", fontsize=9,
            bbox=dict(facecolor="lime", alpha=0.7, pad=1),
        )

    timing = result.get("timing", {})
    breakdown = timing.get("breakdown", {})
    total = timing.get("total_seconds", "?")

    lines = [f"image: {result['image']}", f"status: {result['status']}"]
    if result["status"] == "failed":
        lines.append(f"error: {result.get('error')}")
    else:
        lines.append(f"detections: {len(result['detections'])}")
        lines.append(f'yolo: {breakdown.get("yolo", "-")}s')
        lines.append(f'depth: {breakdown.get("depth", "-")}s')
        lines.append(f'gemini(ai_scene): {breakdown.get("ai_scene", "-")}s')
        lines.append(f'speech: {breakdown.get("speech", "-")}s')
        lines.append(f"total: {total}s")

        scene = result.get("scene_analysis", {}) or {}
        scene_desc = scene.get("analysis", {}).get("description")
        scene_status = scene.get("status")  # "success" = มาจาก gemini จริง, "fallback" = gemini ไม่ตอบ/ปิดอยู่
        if scene_desc:
            lines.append(f"gemini description ({scene_status}): {scene_desc}")
        else:
            lines.append("gemini description: (ไม่มีข้อมูล)")

        if result.get("speech_text"):
            lines.append(f'speech_text: {result["speech_text"]}')

    ax.set_title("\n".join(lines), fontsize=9, loc="left")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="โฟลเดอร์ที่เก็บรูปทดสอบ")
    ap.add_argument("--base-url", default="http://localhost:8000", help="URL ของ API server")
    ap.add_argument("--no-speech", action="store_true", help="ปิด gemini/tts (enable_speech=false)")
    ap.add_argument("--no-show", action="store_true", help="ไม่ต้องเปิดหน้าต่างแสดงรูป (เก็บแค่ CSV)")
    ap.add_argument("--csv-out", default="test_results.csv", help="ไฟล์ CSV ที่จะเซฟสรุป")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"ไม่พบโฟลเดอร์: {folder}")

    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise SystemExit(f"ไม่พบรูปในโฟลเดอร์: {folder}")

    print(f"พบ {len(images)} รูป กำลังทดสอบกับ {args.base_url} ...")

    # เช็คว่า server ตื่นอยู่ก่อน
    try:
        h = requests.get(f"{args.base_url}/health", timeout=5)
        print("health:", h.json())
    except Exception as e:
        raise SystemExit(f"เชื่อมต่อ server ไม่ได้: {e}")

    all_results = []
    csv_rows = []

    for img_path in tqdm(images, desc="testing"):
        try:
            result = test_one_image(args.base_url, img_path, enable_speech=not args.no_speech)
        except Exception as e:
            result = {
                "image": img_path.name, "status": "error",
                "error": str(e), "detections": [], "timing": {},
            }
        all_results.append(result)

        breakdown = result.get("timing", {}).get("breakdown", {})
        scene = result.get("scene_analysis", {}) or {}
        scene_desc = scene.get("analysis", {}).get("description", "")
        scene_status = scene.get("status", "")

        csv_rows.append({
            "image": result["image"],
            "status": result["status"],
            "num_detections": len(result.get("detections", [])),
            "labels": ";".join(d["label"] for d in result.get("detections", [])),
            "yolo_sec": breakdown.get("yolo"),
            "depth_sec": breakdown.get("depth"),
            "gemini_sec": breakdown.get("ai_scene"),
            "speech_sec": breakdown.get("speech"),
            "total_sec": result.get("timing", {}).get("total_seconds"),
            "gemini_status": scene_status,          # success = gemini ตอบจริง / fallback = ไม่ได้ใช้ gemini
            "gemini_description": scene_desc,        # ข้อความคำอธิบายฉากดิบจาก gemini
            "speech_text": result.get("speech_text"),
            "error": result.get("error", ""),
        })

        # print ข้อความ gemini ให้เห็นทันทีทาง terminal ด้วย (กันกรณี font ไทยในรูปเพี้ยน)
        print(f"\n[{result['image']}] gemini({scene_status}): {scene_desc or '(ไม่มีข้อมูล)'}")

    # เซฟ CSV
    with open(args.csv_out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n✓ เซฟสรุปที่ {args.csv_out}")

    # สรุปรวมเวลาเฉลี่ย
    completed = [r for r in csv_rows if r["status"] == "completed"]
    if completed:
        def avg(key):
            vals = [r[key] for r in completed if r[key] is not None]
            return sum(vals) / len(vals) if vals else None

        print("\n--- สรุปเฉลี่ย (เฉพาะที่ completed) ---")
        print(f"จำนวนรูป completed: {len(completed)}/{len(csv_rows)}")
        print(f"yolo เฉลี่ย: {avg('yolo_sec')}")
        print(f"depth เฉลี่ย: {avg('depth_sec')}")
        print(f"gemini เฉลี่ย: {avg('gemini_sec')}")
        print(f"speech เฉลี่ย: {avg('speech_sec')}")
        print(f"total เฉลี่ย: {avg('total_sec')}")

    # แสดงรูปทีละรูป
    if not args.no_show:
        for img_path, result in zip(images, all_results):
            if result["status"] in ("completed", "failed"):
                show_result(img_path, result)


if __name__ == "__main__":
    main()