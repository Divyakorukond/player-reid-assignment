import cv2
from ultralytics import YOLO
import numpy as np
from scipy.spatial import distance
import os
import csv

# Load YOLOv11 model
if not os.path.exists('yolov11.pt'):
    raise FileNotFoundError("Model file not found.")
if not os.path.exists('15sec_input_720p.mp4'):
    raise FileNotFoundError("Video file not found.")

model = YOLO('yolov11.pt')
print("Class names:", model.names)  # Verify class label mapping
# Find the class index for 'person' or 'player'
player_class_idx = None
for idx, name in model.names.items():
    if name.lower() in ["person", "player"]:
        player_class_idx = idx
        break
if player_class_idx is None:
    raise ValueError("Could not find 'person' or 'player' class in model.names")
print(f"Using class index {player_class_idx} for 'player'")

cap = cv2.VideoCapture('15sec_input_720p.mp4')

# Output video writer setup
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_tracking.mp4', fourcc, fps, (width, height))

# CSV writer setup
csv_file = open('tracking_output.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'id', 'x1', 'y1', 'x2', 'y2'])

player_tracks = []
next_id = 0
# Adaptive max_distance: 5% of frame width
max_distance = int(0.05 * width)

# Tunable appearance threshold
appearance_threshold = 0.7

# Histogram parameters with bounding check
def get_histogram(frame, box):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

frame_id = 0
confidence_threshold = 0.3
N = 10  # Max frames to keep a track alive without assignment

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    clss = results[0].boxes.cls.cpu().numpy()

    # Filter players using verified class index
    player_boxes = [
        box for box, cls, conf in zip(boxes, clss, confs)
        if int(cls) == player_class_idx and conf > confidence_threshold
    ]
    centroids = [get_centroid(box) for box in player_boxes]
    histograms = [get_histogram(frame, box) for box in player_boxes]

    # Reset all tracks to unassigned
    for track in player_tracks:
        track['assigned'] = False

    # Match centroids to tracks (first by distance, then by appearance)
    for box, centroid, hist in zip(player_boxes, centroids, histograms):
        min_dist = float('inf')
        min_j = -1
        for j, track in enumerate(player_tracks):
            dist = distance.euclidean(centroid, track['centroid'])
            if dist < min_dist and dist < max_distance and not track['assigned']:
                min_dist = dist
                min_j = j
        if min_j >= 0:
            # Centroid match
            player_tracks[min_j]['centroid'] = centroid
            player_tracks[min_j]['hist'] = hist
            player_tracks[min_j]['assigned'] = True
            player_id = player_tracks[min_j]['id']
        else:
            # Try appearance-based re-ID for unassigned tracks
            best_score = 0.0
            best_k = -1
            for k, track in enumerate(player_tracks):
                if not track['assigned'] and track['last_seen'] > 0 and track['hist'] is not None and hist is not None:
                    score = cv2.compareHist(track['hist'], hist, cv2.HISTCMP_CORREL)
                    if score > best_score and score > appearance_threshold:
                        best_score = score
                        best_k = k
            if best_k >= 0:
                player_tracks[best_k]['centroid'] = centroid
                player_tracks[best_k]['hist'] = hist
                player_tracks[best_k]['assigned'] = True
                player_tracks[best_k]['last_seen'] = 0
                player_id = player_tracks[best_k]['id']
            else:
                # New player
                player_tracks.append({
                    'id': next_id,
                    'centroid': centroid,
                    'assigned': True,
                    'last_seen': 0,
                    'hist': hist
                })
                player_id = next_id
                next_id += 1

        # Draw bounding box and ID
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID: {player_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        # Write to CSV
        csv_writer.writerow([frame_id, player_id, x1, y1, x2, y2])

    # Update last_seen
    for track in player_tracks:
        if not track['assigned']:
            track['last_seen'] += 1
        else:
            track['last_seen'] = 0

    # Prune inactive tracks
    player_tracks = [t for t in player_tracks if t['last_seen'] < N]

    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
