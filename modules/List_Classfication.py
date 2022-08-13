# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:16:25 2022

@author: ELKA
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:07:06 2022

@author: ELKA
"""

import sklearn
import matplotlib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import pickle
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB



def classificationModule(userArray):
    df = pd.read_csv("data/Car_results.csv")
    #copy the data
    df2 = df.copy()

    #change the names of columns from hebrew to english
    df2.rename(columns = {'מהו גילך?':'Age', 'מגדר?':'Gender','מה מספר המושבים ברכב הנוכחי?':'Number_of_seats', 'סוג רישיון של הרכב שלך?': 'License_type','תיבת הילוכים ברכב?':'Gearbox','יצרן של הרכב':'Manufacturer','דגם הרכב':'Model','תת דגם הרכב':'Sub-model','שנת ייצור של הרכב':'Year','סוג מנוע':'Engine_type','נפח מנוע':'Engine_capacity','מחירון הרכב':'Price','הנעה':'Drivetrain','קטגורית הרכב':'Category','מרכב הרכב':'Body','סוג נסיעות שלך?':'Drive_type','האם הרכב משמש אותך לעבודה: ואם כן, לאיזה מטרות משמש?':'Work_car','מספר ילדים?':'Kids','אזור מגורים?':'Living_area','כל כמה זמן את/ה נוהג/ת להחליף רכב?':'Change_cars','מה סיבת ההחלפה בכל פעם?':'Reason','האם תרצה/י לבחור ברכב חשמלי בעתיד?':'Electric_car','מה הפרמטרים החשובים לך בבחירת סוג הרכב?':'Parameters'}, inplace = True)

    #delete the column of time
    #df2 = df2.drop("Time",axis=1)

    #df2.info()
    #df2.isna().sum()

    # Data Frame of Car data
    car_data = df2[['Number_of_seats','License_type','Gearbox','Manufacturer','Model','Sub-model','Year',
                    'Engine_type','Engine_capacity','Price','Drivetrain','Category','Body']]

    # Data Frame of the Data i need
    data_f_c = df2[['Age','Gender','Kids','Living_area','Drive_type','Work_car'
                    ,'Change_cars','Reason','Electric_car','Parameters','Category']]


    #Changing the categoty to English(for my convenience)
    data_f_c["Category"] = data_f_c["Category"].replace(["משפחתי","ג'יפ","קרוסאובר","יוקרה","מיני","מיניוואן","מנהלים","ספורט","מסחרי","טנדר"],['Family','Jeep','Crossover','Luxury','Mini','Minivan','Managers','Sports','Commercial','Tender'])

    #counting the categoty
    #data_f_c['Category'].value_counts()
    #Plot for category
    #data_f_c['Category'].value_counts().plot(kind='bar')
    data_f_c['Parameters'].value_counts()


    # Data Frame for features and target
    X = data_f_c.drop(['Category'],axis=1)
    Y = data_f_c['Category']

    #####################################  Label Encoder #############################

    le = sklearn.preprocessing.LabelEncoder()
    
    #For Age
    Age = le.fit_transform(list(X['Age']))
    Age = le.fit(X['Age'])
    list(le.classes_)
    Age = le.transform(X['Age'])
    #New Data frame of Age as string and Age after label Encoder(for the input later)
    age_str = pd.DataFrame(list(le.inverse_transform(Age)))
    age_str.columns = ['str']
    Age2 = pd.DataFrame(Age)
    Age2.columns = ['num']
    res_age = pd.concat([age_str, Age2], axis=1)


    #For Gender
    Gender = le.fit_transform(list(X['Gender']))
    Gender = le.fit(X['Gender'])
    list(le.classes_)
    Gender = le.transform(X['Gender'])
    #New Data frame of Gender as string and Gender after label Encoder(for the input later)
    gender_str = pd.DataFrame(list(le.inverse_transform(Gender)))
    gender_str.columns = ['str']
    Gender2 = pd.DataFrame(Gender)
    Gender2.columns = ['num']
    res_Gender = pd.concat([gender_str, Gender2], axis=1)

    #For Kids
    Kids = le.fit_transform(list(X['Kids']))
    Kids = le.fit(X['Kids'])
    list(le.classes_)
    Kids = le.transform(X['Kids'])
    #New Data frame of Kids as string and Kids after label Encoder(for the input later)
    kids_str = pd.DataFrame(list(le.inverse_transform(Kids)))
    kids_str.columns = ['str']
    Kids2 = pd.DataFrame(Kids)
    Kids2.columns = ['num']
    res_kids = pd.concat([kids_str, Kids2], axis=1)

    #For Living_area
    Living_area = le.fit_transform(list(X['Living_area']))
    Living_area = le.fit(X['Living_area'])
    list(le.classes_)
    Living_area = le.transform(X['Living_area'])
    #New Data frame of Living_area as string and Living_area after label Encoder(for the input later)
    living_area_str = pd.DataFrame(list(le.inverse_transform(Living_area)))
    living_area_str.columns = ['str']
    Living_area2 = pd.DataFrame(Living_area)
    Living_area2.columns = ['num']
    res_living_area = pd.concat([living_area_str, Living_area2], axis=1)

    #For Drive_type
    Drive_type = le.fit_transform(list(X['Drive_type']))
    Drive_type = le.fit(X['Drive_type'])
    list(le.classes_)
    Drive_type = le.transform(X['Drive_type'])
    #New Data frame of Drive_type as string and Drive_type after label Encoder(for the input later)
    drive_type_str = pd.DataFrame(list(le.inverse_transform(Drive_type)))
    drive_type_str.columns = ['str']
    Drive_type2 = pd.DataFrame(Drive_type)
    Drive_type2.columns = ['num']
    res_drive_type = pd.concat([drive_type_str, Drive_type2], axis=1)

    #For Work_car
    Work_car = le.fit_transform(list(X['Work_car']))
    Work_car = le.fit(X['Work_car'])
    list(le.classes_)
    Work_car = le.transform(X['Work_car'])
    #New Data frame of Work_car as string and Work_car after label Encoder(for the input later)
    work_car_str = pd.DataFrame(list(le.inverse_transform(Work_car)))
    work_car_str.columns = ['str']
    Work_car2 = pd.DataFrame(Work_car)
    Work_car2.columns = ['num']
    res_work_car = pd.concat([work_car_str, Work_car2], axis=1)

    #For Change_cars
    Change_cars = le.fit_transform(list(X['Change_cars']))
    Change_cars = le.fit(X['Change_cars'])
    list(le.classes_)
    Change_cars = le.transform(X['Change_cars'])
    #New Data frame of Change_cars as string and Change_cars after label Encoder(for the input later)
    change_cars_str = pd.DataFrame(list(le.inverse_transform(Change_cars)))
    change_cars_str.columns = ['str']
    Change_cars2 = pd.DataFrame(Change_cars)
    Change_cars2.columns = ['num']
    res_change_cars = pd.concat([change_cars_str, Change_cars2], axis=1)

    #For Reason
    Reason = le.fit_transform(list(X['Reason']))
    Reason = le.fit(X['Reason'])
    list(le.classes_)
    Reason = le.transform(X['Reason'])
    #New Data frame of Reason as string and Reason after label Encoder(for the input later)
    reason_str = pd.DataFrame(list(le.inverse_transform(Reason)))
    reason_str.columns = ['str']
    Reason2 = pd.DataFrame(Reason)
    Reason2.columns = ['num']
    res_reason = pd.concat([reason_str, Reason2], axis=1)

    #For Electric_car
    Electric_car = le.fit_transform(list(X['Electric_car']))
    Electric_car = le.fit(X['Electric_car'])
    list(le.classes_)
    Electric_car = le.transform(X['Electric_car'])
    #New Data frame of Electric_car as string and Electric_car after label Encoder(for the input later)
    electric_car_str = pd.DataFrame(list(le.inverse_transform(Electric_car)))
    electric_car_str.columns = ['str']
    Electric_car2 = pd.DataFrame(Electric_car)
    Electric_car2.columns = ['num']
    res_electric_car = pd.concat([electric_car_str, Electric_car2], axis=1)

    #For Parameters
    Parameters = le.fit_transform(list(X['Parameters']))
    Parameters = le.fit(X['Parameters'])
    list(le.classes_)
    Parameters = le.transform(X['Parameters'])
    #New Data frame of Parameters as string and Parameters after label Encoder(for the input later)
    parameters_str = pd.DataFrame(list(le.inverse_transform(Parameters)))
    parameters_str.columns = ['str']
    Parameters2 = pd.DataFrame(Parameters)
    Parameters2.columns = ['num']
    res_parameters = pd.concat([parameters_str, Parameters2], axis=1)


    #For Category
    le_Category = LabelEncoder()

    ##########################################################################################

    X_n = list(zip(Age,Gender,Kids,Living_area,Drive_type,Work_car,Change_cars,Reason,Electric_car,Parameters))

    #Mapping the Category
    Y_n = Y.map({"Family": 0, "Jeep": 1,"Crossover": 2
                ,"Mini": 3, "Minivan": 4,"Managers": 5, "Sports": 6
                ,"Commercial": 7})


    #Giving names(for the out put)
    class_names = ["Family", "Jeep","Crossover"
                ,"Mini", "Minivan","Managers", "Sports"
                ,"Commercial"]

    '''
    ################################# Decision Tree #########################################
    X_n_train, X_n_test, Y_n_train, Y_n_test = train_test_split(X_n, Y_n, test_size= 0.10, train_size = 0.90)
    DT = DecisionTreeClassifier(criterion = 'gini',max_depth= 17)
    DT.fit(X_n_train,Y_n_train)
    print(DT.score(X_n_test, Y_n_test))
    DtResults = DT.predict(X_n_test)

    #Confusion_Matrix_Tree
    condusion_matrix_tree = metrics.confusion_matrix(Y_n_test,DtResults)
    print(condusion_matrix_tree)
    print(metrics.classification_report(Y_n_test,DtResults))


    cnt=0
    print(len(DtResults))
    for i in range(len(DtResults)):
        print("Actual: " + class_names[(DtResults[i])])
        print("Prediction: " + class_names[np.argmax(DtResults[i])]+ "\n")
        if class_names[(DtResults[i])] == class_names[np.argmax(DtResults[i])]:
            cnt = cnt+1
    print(cnt)

    ################ Cross Validation for Decision Tree
    param_dist = {"max_depth":[3,None],
                "min_samples_leaf": randint(1,9),
                "criterion":["gini","entropy"]}

    tree = DecisionTreeClassifier()
    tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
    tree_cv.fit(X_n_train,Y_n_train)
    print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
    print("Best score is {}".format(tree_cv.best_score_))
    #'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 4
    #0.48

    cv = ShuffleSplit(n_splits=5, test_size =0.10 , random_state=0)
    cross_val_score(DecisionTreeClassifier(), X_n, Y_n, cv=cv)


    ################# max_depth for Decision Tree
    parameters = {'max_depth':range(3,20)}
    clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
    clf.fit(X=X_n, y=Y_n)
    tree_model = clf.best_estimator_
    print (clf.best_score_, clf.best_params_) 
    #0.46616541353383456 {'max_depth': 17}
    '''
    ####################################################################################################


    ############################ RandomForestClassifier ###################################

    X_n_train, X_n_test, Y_n_train, Y_n_test = train_test_split(X_n, Y_n, test_size= 0.10, train_size = 0.90)
    Ra_C = RandomForestClassifier(criterion = 'gini', max_features=3)
    Ra_C.fit(X_n_train,Y_n_train)


    '''
    Y_Pred = Ra_C.predict(X_n_test)
    print(confusion_matrix(Y_n_test, Y_Pred))
    print(metrics.classification_report(Y_n_test, Y_Pred))
    print(accuracy_score(Y_n_test, Y_Pred))
    '''

    '''
    #saving the best model
    best = 0
    for i in range(500):
        Ra_C = RandomForestClassifier(criterion = 'gini',max_features=3)
        Ra_C.fit(X_n_train,Y_n_train)
        acc = Ra_C.score(X_n_test,Y_n_test)
        print(acc)
        
        if acc > best:
            best = acc
            with open("Carmodel.pickle", "wb") as f:
                pickle.dump(Ra_C, f)
                
    print("The best acc is: " ,best)
    #The best acc is: 0.775
    '''
    # pickle_in = open("Carmodel.pickle", "rb")
    # Ra_C = pickle.load(pickle_in)


    '''
    prediction2 = Ra_C.predict(X_n_test)
    print(Ra_C.score(X_n_test, Y_n_test))
    print(confusion_matrix(Y_n_test, prediction2))
    print(metrics.classification_report(Y_n_test, prediction2))
    print(accuracy_score(Y_n_test, prediction2))
    '''

    '''
    # הדפסה של כמה הוא צדק בחיזוי לעומת האמיתי
    cnt=0
    print(len(prediction2))
    for i in range(len(prediction2)):
        print("Actual: " + class_names[(prediction2[i])])
        print("Prediction: " + class_names[np.argmax(prediction2[i])]+ "\n")
        if class_names[(prediction2[i])] == class_names[np.argmax(prediction2[i])]:
            cnt = cnt+1
    print(cnt)
    '''

    '''
    ################ Cross Validation for Random Forest
    cv = ShuffleSplit(test_size =0.10)
    cross_val_score(RandomForestClassifier(), X_n, Y_n, cv=cv)
    '''

    
    ###################### convert input to numbers
    put2 = []
    pd.options.mode.chained_assignment = None
    for i in range(1):
        age_of_user = res_age.loc[res_age['str'] == userArray[0], 'num'].iloc[0]
        put2.append(age_of_user)
        
        gender_of_user = res_Gender.loc[res_Gender['str'] == userArray[1], 'num'].iloc[0]
        put2.append(gender_of_user)
        
        kids_of_user = res_kids.loc[res_kids['str'] == userArray[2], 'num'].iloc[0]
        put2.append(kids_of_user)
        
        living_area_of_user = res_living_area.loc[res_living_area['str'] == userArray[3], 'num'].iloc[0]
        put2.append(living_area_of_user)
        
        drive_type_of_user = res_drive_type.loc[res_drive_type['str'] == userArray[4], 'num'].iloc[0]
        put2.append(drive_type_of_user)
        
        work_car_of_user = res_work_car.loc[res_work_car['str'] == userArray[5], 'num'].iloc[0]
        put2.append(work_car_of_user)
        
        change_cars_of_user = res_change_cars.loc[res_change_cars['str'] == userArray[6], 'num'].iloc[0]
        put2.append(change_cars_of_user)
        
        reason_of_user = res_reason.loc[res_reason['str'] == userArray[7], 'num'].iloc[0]
        put2.append(reason_of_user)
        
        electric_car_of_user = res_electric_car.loc[res_electric_car['str'] == userArray[8], 'num'].iloc[0]
        put2.append(electric_car_of_user)
        
        parameters_of_user = res_parameters.loc[res_parameters['str'] == userArray[9], 'num'].iloc[0]
        put2.append(parameters_of_user)
            
    # print(put2)

    ######################## classification to the input of the user
    features = np.array([put2])
    # print(features)
    prediction = Ra_C.predict(features)
    num_pre = prediction
    # print(class_names[int(num_pre)])

    ######### Out put for the user
    #הדפסה של הרכבים תחת אותה קטגוריה
    car_data["Category"] = car_data["Category"].replace(["משפחתי","ג'יפ","קרוסאובר","יוקרה","מיני","מיניוואן","מנהלים","ספורט","מסחרי","טנדר"],['Family','Jeep','Crossover','Luxury','Mini','Minivan','Managers','Sports','Commercial','Tender'])
    temp = car_data.loc[car_data['Category'] == class_names[int(num_pre)]]
    temp2 = temp[['Manufacturer','Model','Year','Price']]
    # print(temp2)
    return temp2
    
    #print(temp2.loc[(temp2['Price'] >= min_price) & (temp2['Price'] <= max_price)][:5])

    #####################################################################################


    '''
    ##########################  NAIVE BAYES #######################################

    NB = GaussianNB()
    NB.fit(X_n_train,Y_n_train)
    NbResults = NB.predict(X_n_test)
    print(metrics.classification_report(Y_n_test, NbResults))
    #0.34

    cnt=0
    print(len(NbResults))
    for i in range(len(NbResults)):
        print("Actual: " + class_names[(NbResults[i])])
        print("Prediction: " + class_names[np.argmax(NbResults[i])]+ "\n")
        if class_names[(NbResults[i])] == class_names[np.argmax(NbResults[i])]:
            cnt = cnt+1
    print(cnt)
    ###############################################################################
    '''
    '''
    #feature_selection 
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_n_train,Y_n_train)
    X_train_fs = fs.transform(X_n_train)
    X_test_fs = fs.transform(X_n_test)

    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))

    #Feature 0 : 46.117027 AGE
    #Feature 1: 35.579329  GENDER
    #Feature 2: 157.130216 KIDS
    #Feature 3: 16.234768 LIVING_AREA
    #Feature 4: 9.902699  DRIVE_TYPE
    #Feature 5: 14.018730  WORK_CAR
    #Feature 6: 10.570679 CHANGE_CARS
    #Feature 7: 31.893936 REASON
    #Feature 8: 40.106328 ELECTRIC_CAR
    #Feature 9: 215.015492 PARAMETERS
    '''
    