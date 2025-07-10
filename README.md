# 🏃‍♀️ Player Re-Identification using YOLOv11

This project solves the problem of **player tracking and re-identification** in sports video footage using **YOLOv11**, OpenCV, and histogram-based appearance matching. It ensures that each player is consistently identified across frames, even if they temporarily leave and re-enter the scene.

---

## 📁 Files

- `sample.py` – Main Python script implementing the tracking and re-ID logic.
- `output_tracking.mp4` – (Generated) Video with bounding boxes and player IDs.
- `tracking_output.csv` – (Generated) CSV file containing `[frame, id, x1, y1, x2, y2]`.

---

## 🧠 Features

- ✅ YOLOv11-based player detection
- ✅ Centroid-based spatial tracking
- ✅ Histogram-based appearance re-identification
- ✅ Adaptive distance threshold
- ✅ Handles temporary occlusion and exits
- ✅ Output to both annotated video and structured CSV

---

## 🧩 Dependencies

Install the required Python packages:

```bash
pip install ultralytics opencv-python numpy scipy
