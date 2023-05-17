from django.shortcuts import render, redirect
from django.http import HttpResponse
from New_way import models
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,logout,login as auth_login
from django.http import HttpResponse
from New_way.models import signup
# Create your views here.
def New_way(request):
    return render(request,'home.html')
def home(request):
    return render(request,'home.html')
def employe(request):
    return render(request,'employe.html')
def user(request):
    return render(request,'user.html')

def eresult(request):
    return render(request, 'employeeresult.html')


    
def login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        try:
            udata = User.objects.get(email=email)
            username = udata.username
            print(username)
            authuser = authenticate(request, username=username, password=password)
            if authuser is not None:
                auth_login(request, authuser)
                return redirect('home')
            else:
                error = "Invalid Data"
        except:
            error = "User not found"
    else:
        error = ""

    return render(request, 'user.html', {'error': error})

def logouts(request):
    logout(request)
    return redirect('home')
def job_hiring(request):
    if request.method == 'POST':
        obj = models.hiring()
        obj.name = request.POST['ename']
        obj.password = request.POST['epass']
        obj.location = request.POST['eaddress']
        obj.contactno = request.POST['econno']
        obj.mail = request.POST['eemail']
        obj.qualification = request.POST['equalification']
        obj.jobrole = request.POST['ejob']
        obj.experience = request.POST['eexperience']
        obj.save()
        print('!!!!!!!!!!!!!!')
        return render(request,"employeeresult.html")
    else:
        return render(request,'Job_hiring.html')
# def register(request):
#     return render(request,'Register.html')
def services(request):
    return render(request,'services.html')



def sign(request):
    error=""
    if request.method == 'POST':
        obj = models.signup()
        obj.username = request.POST['signname']
        obj.email = request.POST['signemail']
        obj.password = request.POST['signpassword']
        print('!!!!!!!!!!!!')
        if User.objects.filter(email=obj.email).exists():
            error="email already exists"
        else:
            user=User.objects.create_user(username=obj.username,email=obj.email,password=obj.password)
            user.save()
            obj.save()
            return redirect('login')
    
    return render(request,'signup.html',{'error':error})


    


    
    











def ac(request):
    if request.method == 'POST':
        obj = models.services()
        obj.service_for = request.POST['acserve']
        obj.service_required_on = request.POST['acdate']
        obj.service_required_at = request.POST['acaddress']
        obj.contact = request.POST['accno']
        obj.mail = request.POST['acemail']
        obj.information = request.POST['acaddinf']
        obj.save()
        print('@@@@@@@@@@')
        return render(request, 'result.html')
    else:
        return render(request, 'acservice.html')
    

def fridge(request):
    if request.method == 'POST':
        obj = models.services()
        obj.service_for = request.POST['fridgeserve']
        obj.service_required_on = request.POST['fridgedate']
        obj.service_required_at = request.POST['fridgeaddress']
        obj.contact = request.POST['fridgecno']
        obj.mail = request.POST['fridgeemail']
        obj.information = request.POST['fridgeaddinf']
        obj.save()
        print('@@@@@@@@@@')
        return render(request, 'result.html')
    else:
        return render(request, 'fridgeservice.html')

    


def tv(request):
    if request.method == 'POST':
        obj = models.services()
        obj.service_for = request.POST['tvserve']
        obj.service_required_on = request.POST['tvdate']
        obj.service_required_at = request.POST['tvaddress']
        obj.contact = request.POST['tvcno']
        obj.mail = request.POST['tvemail']
        obj.information = request.POST['tvaddinf']
        obj.save()
        print('@@@@@@@@@@')
        return render(request, 'result.html')
    else:
        return render(request, 'tvservice.html')


def car(request):
    if request.method == 'POST':
        obj = models.services()
        obj.service_for = request.POST['carserve']
        obj.service_required_on = request.POST['cardate']
        obj.service_required_at = request.POST['caraddress']
        obj.contact = request.POST['carcno']
        obj.mail = request.POST['caremail']
        obj.information = request.POST['caraddinf']
        obj.save()
        print('@@@@@@@@@@')
        return render(request, 'result.html')
    else:
        return render(request, 'carservice.html')


def bike(request):
    if request.method == 'POST':
        obj = models.services()
        obj.service_for = request.POST['bikeserve']
        obj.service_required_on = request.POST['bikedate']
        obj.service_required_at = request.POST['bikeaddress']
        obj.contact = request.POST['bikecno']
        obj.mail = request.POST['bikeemail']
        obj.information = request.POST['bikeaddinf']
        obj.save()
        print('@@@@@@@@@@')
        return render(request, 'result.html')
    else:
        return render(request, 'bikeservice.html')
    



def electric(request):
    if request.method == 'POST':
        obj = models.services()
        obj.service_for = request.POST['electricserve']
        obj.service_required_on = request.POST['electricdate']
        obj.service_required_at = request.POST['electricaddress']
        obj.contact = request.POST['electriccno']
        obj.mail = request.POST['electricemail']
        obj.information = request.POST['electricaddinf']
        obj.save()
        print('@@@@@@@@@@')
        return render(request, 'result.html')
    else:
        return render(request, 'electricservice.html')
    


def plumbing(request):
    if request.method == 'POST':
        obj = models.services()
        obj.service_for = request.POST['plumserve']
        obj.service_required_on = request.POST['plumdate']
        obj.service_required_at = request.POST['plumaddress']
        obj.contact = request.POST['plumcno']
        obj.mail = request.POST['plumemail']
        obj.information = request.POST['plumaddinf']
        obj.save()
        print('@@@@@@@@@@')
        return render(request, 'result.html')
    else:
        return render(request, 'plumbingservice.html')
    