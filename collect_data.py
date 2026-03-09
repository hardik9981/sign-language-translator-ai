# collect_data.py
import cv2, os

# --- CHANGE HERE for each sign ---
label = "Drink"        # e.g. "Hello", "Yes", "No" or "A", "B"
num_samples = 1000      # images to collect for this label
# -------------------------------

save_dir = os.path.join("dataset", label)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 's' to start/stop saving, 'q' to quit.")
count = len(os.listdir(save_dir))
saving = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    h, w = frame.shape[:2]
    size = min(h, w) // 2
    cx, cy = w // 2, h // 2
    x1, y1 = cx - size//2, cy - size//2
    x2, y2 = x1 + size, y1 + size
    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,155,0), 2)
    cv2.putText(frame, f"Label:{label} Count:{count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        saving = not saving
    elif key == ord('q'):
        break

    if saving and count < num_samples:
        img = cv2.resize(roi, (424,424))
        cv2.imwrite(os.path.join(save_dir, f"{count:05d}.jpg"), img)
        count += 1

cap.release()
cv2.destroyAllWindows() 