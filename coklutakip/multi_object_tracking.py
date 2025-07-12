import cv2

# 🎯 OpenCV sürümünde yalnızca legacy üzerinden kullanım
TrackerCreate = lambda name: getattr(cv2.legacy, f"Tracker{name}_create")
MultiTracker = cv2.legacy.MultiTracker_create

OPENCV_OBJECT_TRACKERS = {
    "csrt": TrackerCreate("CSRT"),
    "kcf": TrackerCreate("KCF"),
    "boosting": TrackerCreate("Boosting"),
    "mil": TrackerCreate("MIL"),
    "tld": TrackerCreate("TLD"),
    "medianflow": TrackerCreate("MedianFlow"),
    "mosse": TrackerCreate("MOSSE")
}

tracker_name = "medianflow"
trackers = MultiTracker()

video_path = "MOT17-04-DPM.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Video bitti ya da kare alınamadı.")
        break

    frame = cv2.resize(frame, (960, 540))
    success, boxes = trackers.update(frame)

    info = [
        ("Tracker", tracker_name),
        ("Success", "Yes" if success else "No")
    ]
    string_text = " ".join(["{}: {}".format(k, v) for (k, v) in info])

    cv2.putText(frame, string_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("t"):
        box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        if box != (0, 0, 0, 0):
            tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
            trackers.add(tracker, frame, box)
            print(f"📌 Yeni nesne eklendi: {box}")

    elif key == ord("q"):
        print("🚪 Çıkış yapılıyor...")
        break

cap.release()
cv2.destroyAllWindows()
