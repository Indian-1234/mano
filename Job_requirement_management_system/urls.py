"""Job_requirement_management_system URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from New_way import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.New_way, name='home'),
    path('employe',views.employe),
    path('user',views.user),
    path('login',views.login,name='login'),
    path('logout',views.logouts,name="logout"),
    path('signup',views.sign),
    path('Job_hiring',views.job_hiring, name='jobhiring'),
    # path('Job_hiring/Register/',views.register,name='register'),
    path('services',views.services, name='services'),
    path('ac',views.ac),
    path('fridge', views.fridge),
    path('tv', views.tv),
    path('car', views.car),
    path('bike', views.bike),
    path('electric', views.electric),
    path('plumbing', views.plumbing),
    #path('submit',views.submit),
    path('eresult', views.eresult),
    #path('Signup',views.signup)
]
