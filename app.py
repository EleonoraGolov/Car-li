from hashlib import new
import re
from unittest import result
from flask import Flask, render_template , url_for, redirect , request
import pandas as pd
import csv as csv
import numpy as np
from List_Classfication import classificationModule
from googletrans import Translator

app = Flask(__name__,template_folder='templates')
app.config['SECRET_KEY'] = 'secret key'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1
app.config["TEMPLATES_AUTO_RELOAD"] = True

result_arr=[]
userArray=[]

#---Render Home Page---
@app.route('/')
def HomePage():
    return render_template('home.html')
     
#---Render Form Page---
@app.route('/form', methods=['GET','POST'])
def FormPage():
    #set all variables
    result_arr.clear()
    living_place = None
    age = None
    gender = None
    kids = None
    drive_type = None
    work_car = None
    change_cars_of_user = None
    reason_of_user = None
    electric_car_of_user = None
    min_price = None
    max_price = None
    preferences = None
   
    #when submit button is clicked, get all values from user
    
    if request.method == 'POST': 
       
        living_place = request.form['living-place']
        age = request.form['age']
        gender = request.form['gender']
        kids = request.form['kids']
        drive_type = request.form['drive-type']
        work_car = request.form['work-car']  
        change_cars_of_user = request.form['change-cars-of-user']
        reason_of_user = request.form['reason-of-user']
        electric_car_of_user = request.form['electric-car-of-user']
        if request.form['range_min'] <= request.form['range_max']:
            min_price = request.form['range_min']  
        max_price = request.form['range_max'] 
        preferences = request.form.getlist('preferences')
        
        userArray.append(age)
        userArray.append(gender)
        userArray.append(kids)
        userArray.append(living_place)
        userArray.append(drive_type)
        userArray.append(work_car)
        userArray.append(change_cars_of_user)
        userArray.append(reason_of_user)
        userArray.append(electric_car_of_user)
        newPreferences = ' '.join(map(str,preferences)).replace(' ',', ')
        userArray.append(newPreferences)
        print(userArray)
        
        #arr = np.array(['17-27','נקבה','אין','צפון','משולב','לא','מעל 5 שנים','שידרוג רכב','כן','מחיר, אמינות, בטיחות'])
        #arr2 = np.array(['17-27', 'זכר', 'אין', 'צפון', 'משולב', 'הסעת עובדים', 'כל חצי שנה או פחות', 'הרחבת המשפחה', 'לא', 'בטיחות, גודל(גדול)'])
        
        
        
        #put some algorithm and render result page
        return redirect('result')
    else:
        return render_template('form.html',living_place=living_place,age=age,gender=gender,kids=kids,drive_type=drive_type,work_car=work_car,change_cars_of_user=change_cars_of_user,reason_of_user=reason_of_user,min_price=min_price,max_price=max_price,preferences=preferences)

#---Render Result Page---
@app.route('/result', methods=['GET','POST'])
def ResultPage():
    result_arr = classificationModule(np.array(userArray))
    print(result_arr)
    
    data = pd.read_csv('data/data.csv')
    manufacturer_list = result_arr.loc[:,'Manufacturer']
    model_list = result_arr.loc[:,'Model']
    year_list = result_arr.loc[:,'Year']
    price_list = result_arr.loc[:,'Price']
    images_list = []

    # for img in range(len(manufacturer_list)):
    #     images_list.append(Get_Image_From_API(manufacturer_list[img],model_list[img],str(year_list[img])))
    
    #Result Object send to Client side
    cars_data = zip(manufacturer_list,model_list,year_list,price_list)
    userArray.clear()
    return render_template('result.html',cars_data=cars_data)
    


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)



