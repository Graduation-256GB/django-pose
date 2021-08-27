# backend/post/admin.py
from django.contrib import admin
from .models import Exercise, ExerciseSet, Set
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, ExerciseSet, Exercise, ExerciseLog, Set


admin.site.register(Exercise)
admin.site.register(CustomUser)
admin.site.register(ExerciseSet)
admin.site.register(Set)
admin.site.register(ExerciseLog)


class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ['username']
