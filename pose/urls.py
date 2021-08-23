from django.urls import path, include
from pose import views


urlpatterns = [
    path('', views.index, name='index'),
    path('pose_feed', views.pose_feed, name='pose_feed'),
]
