import cv2
import numpy as np

testmode = 1

def make_chunks(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]

cap = cv2.VideoCapture(0)

#TURUNKAN RESOLUSI (PENTING!)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

StepSize = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_h, img_w = frame.shape[:2]

    #Blur lebih ringan
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 100)

    EdgeArray = []

    #GANTI LOOP PIXEL → NUMPY (SUPER CEPAT)
    for j in range(0, img_w, StepSize):
        column = edges[:, j]  # ambil 1 kolom

        # cari index edge dari bawah
        indices = np.where(column[::-1] == 255)[0]

        if len(indices) > 0:
            i = img_h - indices[0] - 1
            EdgeArray.append((j, i))
        else:
            EdgeArray.append((j, 0))

    # 🔥 NAVIGASI
    number_of_chunks = 3
    size = len(EdgeArray) // number_of_chunks
    chunks = make_chunks(EdgeArray, size)

    avg_points = []

    for chunk in chunks[:3]:  # pastikan hanya 3
        xs = [p[0] for p in chunk]
        ys = [p[1] for p in chunk]

        avg_x = int(np.mean(xs))
        avg_y = int(np.mean(ys))

        avg_points.append((avg_x, avg_y))

        if testmode:
            cv2.circle(frame, (avg_x, avg_y), 5, (255, 0, 0), -1)

    forward = avg_points[1]

    if forward[1] > img_h * 0.6:
        left = avg_points[0]
        right = avg_points[2]

        if left[1] < right[1]:
            direction = "LEFT"
        else:
            direction = "RIGHT"
    else:
        direction = "FORWARD"

    print(direction)

    if testmode:
        cv2.putText(frame, direction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        cv2.imshow("Edges", edges)
        cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()