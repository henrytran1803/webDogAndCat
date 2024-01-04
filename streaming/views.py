from django.shortcuts import render
import cv2
from keras.preprocessing.image import load_img
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from django.shortcuts import render, redirect, get_object_or_404
from keras.models import load_model
import numpy as np
from django.http import HttpResponse
import os
# Create your views here.

picture = None
def index(request):
    return render(request, 'streaming/index.html')

# Hàm đọc video từ webcam
def video_stream():
    global picture
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('rtsp://172.20.10.8:8554/mjpeg/1')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.resize(frame, (640, 480))

        _, jpeg = cv2.imencode('.jpg', frame)
        picture = jpeg
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# View chứa hàm streaming
@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(video_stream(), content_type="multipart/x-mixed-replace;boundary=frame")

def classifications(request):
    result = answer()
    context = {'answer': result}
    return render(request, 'streaming/answers.html', context)

def loadmodel():
    try:
        loaded_model = load_model("model/model.h5")
        return loaded_model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None
def answer():
    global picture
    model = loadmodel()
    try:
        if picture is not None and len(picture) > 0:
            img_array = np.frombuffer(picture, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (150, 150))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Dự đoán lớp của hình ảnh
            pred = model.predict(img)

            if pred[0, 0] > 0.5:
                label = 'Chó'
            else:
                label = 'Mèo'

            return label
        else:
            print("Lỗi: Ảnh rỗng hoặc không tồn tại.")
            return None  # Hoặc có thể trả về giá trị khác để xử lý lỗi
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        return None  # Hoặc có thể trả về giá trị khác để xử lý lỗi
