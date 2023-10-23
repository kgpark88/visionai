import cv2

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
while True:
    # capture video frame
    ret, frame = video_capture.read()
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(
        image_gray, minSize=(100, 100), minNeighbors=5
    )

    # draw a rectangle around the faces
    for x, y, w, h in detections:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display the resulting frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the video capture
video_capture.release()
cv2.destroyAllWindows()
