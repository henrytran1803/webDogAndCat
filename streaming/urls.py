from django.urls import path, include
from streaming import views


urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('classifications/', views.classifications, name='classifications'),


    ]