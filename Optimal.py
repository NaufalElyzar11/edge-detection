import cv2
import numpy as np

testmode = 0  # Matikan saat deploy robot!

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Kurangi buffer lag

StepSize = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_h, img_w = frame.shape[:2]

    # [OPTIMASI 1] Langsung ke grayscale, Canny tidak butuh warna
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # [OPTIMASI 2] Blur lebih kecil + blur di grayscale (lebih cepat)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blur, 50, 100)

    # [OPTIMASI 3] Proses semua kolom sekaligus pakai numpy (tanpa Python for loop)
    cols = np.arange(0, img_w, StepSize)
    selected = edges[:, cols]  # shape: (img_h, n_cols)

    # Flip vertikal, cari index pertama yg 255 tiap kolom
    flipped = selected[::-1, :]
    row_indices = np.argmax(flipped == 255, axis=0)  # index dari bawah
    has_edge = np.any(flipped == 255, axis=0)

    edge_y = np.where(has_edge, img_h - row_indices - 1, 0)
    EdgeArray = list(zip(cols, edge_y))  # [(x, y), ...]

    # [OPTIMASI 4] numpy array split gantikan make_chunks
    n = len(EdgeArray) // 3
    chunks = [EdgeArray[:n], EdgeArray[n:2*n], EdgeArray[2*n:]]

    avg_points = []
    for chunk in chunks:
        arr = np.array(chunk)
        avg_x = int(np.mean(arr[:, 0]))
        avg_y = int(np.mean(arr[:, 1]))
        avg_points.append((avg_x, avg_y))

    forward = avg_points[1]

    if forward[1] > img_h * 0.6:
        left, right = avg_points[0], avg_points[2]
        direction = "LEFT" if left[1] < right[1] else "RIGHT"
    else:
        direction = "FORWARD"

    print(direction)

    if testmode:
        for pt in avg_points:
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)
        cv2.putText(frame, direction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Edges", edges)
        cv2.imshow("Frame", frame)

    # [OPTIMASI 5] waitKey lebih tinggi hemat CPU saat testmode
    key = cv2.waitKey(10 if testmode else 1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()