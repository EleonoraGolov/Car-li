from dataclasses import replace
from unittest import result
from flask import Flask, render_template, url_for, redirect, request
import pandas as pd
import csv as csv
import numpy as np
from modules.List_Classfication import classificationModule
from googletrans import Translator

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret key'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1
app.config["TEMPLATES_AUTO_RELOAD"] = True

result_arr = []
userArray = []
pricesRange = []

# ---Render Home Page---


@app.route('/')
def HomePage():
    return render_template('home.html')

# ---Render Form Page---


@app.route('/form', methods=['GET', 'POST'])
def FormPage():
    # set all variables
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
    min_price = 0
    max_price = 0
    preferences = None

    # when submit button is clicked, get all values from user

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
        if request.form['range_min'] < request.form['range_max']:
            min_price = request.form['range_min']
        max_price = request.form['range_max']
        pricesRange.append(min_price)
        pricesRange.append(max_price)
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
        newPreferences = ' '.join(map(str, preferences)).replace(' ', ', ')
        userArray.append(newPreferences)

        # put some algorithm and render result page
        return redirect('result')
    else:
        return render_template('form.html', living_place=living_place, age=age, gender=gender, kids=kids, drive_type=drive_type, work_car=work_car, change_cars_of_user=change_cars_of_user, reason_of_user=reason_of_user, min_price=min_price, max_price=max_price, preferences=preferences)

# ---Render Result Page---


@app.route('/result', methods=['GET', 'POST'])
def ResultPage():
    cars_data=[]
    result_arr = classificationModule(np.array(userArray))
   
    for index, row in result_arr.iterrows():
        if (float(row['Price'].replace(",","")) <= float(pricesRange[1])) & (float(row['Price'].replace(",","")) >= float(pricesRange[0])):
            cars_data.append(row)
            if len(cars_data) == 5:
                break
            

    images_list = []

    # for img in range(len(manufacturer_list)):
    #     images_list.append(Get_Image_From_API(manufacturer_list[img],model_list[img],str(year_list[img])))

    # Result Object send to Client side
    
    
    
    # for i in cars_data:
    #     if float(i[3].replace(",","")) < float(pricesRange[1]):
    #         print(i)
    
    userArray.clear()
    pricesRange.clear()
    return render_template('result.html', cars_data=cars_data)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
