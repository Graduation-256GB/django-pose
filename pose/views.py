from django.shortcuts import render
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from pose.camera import PoseWebCam


def index(request):
    return render(request, 'pose/home.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
# Create your views here.


# make your set 페이지

def set_create(request):
    pass


def set_detail(request):
    pass


def set_delete(request):
    pass


def set_list(request):
    pass


# training 페이지
def train_get(request, pk):
    pass


def train_start(request, pk):
    return StreamingHttpResponse(gen(PoseWebCam()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def train_result(request, pk):
    pass
