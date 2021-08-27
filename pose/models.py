from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db.models.fields import CharField
from django.utils import timezone


class Exercise(models.Model):
    name = models.CharField(max_length=50)
    calories = models.IntegerField()
    img = models.ImageField(null=True, blank=True)
    url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)  # 추가된 시간
    # updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class CustomUser(AbstractUser):
    # Any extra fields would go here
    def __str__(self):
        return self.username


class Set(models.Model):
    name = models.CharField(max_length=40)
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE)
    date = models.DateField()
    type = models.Choices(
        ('상체', '상체')
        ('하체', '하체')
        ('기타', '기타')
    )
    created_at = models.DateTimeField(auto_now_add=True)
    # updated_at = models.DateTimeField(auto_now=True)


class ExerciseSet(models.Model):
    exercise = models.ForeignKey(
        Exercise, on_delete=models.SET_NULL, null=True)  # 어떤 운동인지
    set = models.ForeignKey(Set, on_delete=models.CASCADE)  # 어떤 세트에 속하는지
    set_num = models.IntegerField()    # 세트 내에서 운동 순서, 몇번째 운동인지
    set_count = models.IntegerField()   # 세트 내에서 정해진 운동 횟수
    created_at = models.DateTimeField(auto_now_add=True)
    # updated_at = models.DateTimeField(auto_now=True)


class ExerciseLog(models.Model):
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE)
    correct_count = models.IntegerField(default=0)  # 사용자가 운동을 정확하게 했을 때의 카운트
    fail_count = models.IntegerField(default=0)  # 사용자가 운동을 실패했을 때의 카운트
    # set = models.ForeignKey(Set, on_delete=models.CASCADE)
    # set_num = models.IntegerField()
    set_exercise = models.ForeignKey(ExerciseSet, on_delete=models.PROTECT)
    # created_at = models.DateTimeField(auto_now_add=True)
    # updated_at = models.DateTimeField(auto_now=True)
    time_started = models.DateTimeField(blank=True, null=True)
    time_finished = models.DateTimeField(blank=True, null=True)
