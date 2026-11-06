import cv2
from flask import Flask, Response

app = Flask(__name__)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Could not open laptop webcam")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 15)

def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

@app.route("/")
def index():
    return """
    <html>
    <body style="background:#111;color:white;text-align:center">
    <h2>Laptop Webcam Stream</h2>
    <img src="/video_feed" width="640">
    </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    print("====================================")
    print("Laptop webcam stream is running")
    print("Test: http://127.0.0.1:5000")
    print("====================================")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
