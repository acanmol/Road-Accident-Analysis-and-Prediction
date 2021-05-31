# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import joblib

plt.style.use('ggplot')

# Importing the dataset
accidents = pd.read_csv('DataSets/Road Safety Data - Accidents 2019.csv', index_col='Accident_Index', low_memory=False)
casualties = pd.read_csv('DataSets/Road Safety Data - Casualties 2019.csv', index_col='Accident_Index', low_memory=False)
vehicles = pd.read_csv('DataSets/Road Safety Data - Vehicles 2019.csv', index_col='Accident_Index', low_memory=False)

accidents.drop(['Location_Easting_OSGR', 'Location_Northing_OSGR', 'LSOA_of_Accident_Location',
                'Junction_Control', '2nd_Road_Class'], axis=1, inplace=True)
casualties.drop('Pedestrian_Road_Maintenance_Worker', axis=1, inplace=True)

accidents['Date_time'] = accidents['Date'] + ' ' + accidents['Time']

for col in accidents.columns:
    accidents = (accidents[accidents[col] != -1])

for col in casualties.columns:
    casualties = (casualties[casualties[col] != -1])
    
for col in vehicles.columns:
    vehicles = (vehicles[vehicles[col] != -1])

accidents['Date_time'] = pd.to_datetime(accidents.Date_time)
accidents.drop(['Date', 'Time'], axis=1, inplace=True)
accidents = accidents.join(vehicles, how='outer')

accidents.dropna(inplace=True)

# plot for accidents on the days of a week
plt.figure(figsize=(12, 6))
accidents.Date_time.dt.dayofweek.hist(bins=7, rwidth=0.55, color='red')
plt.title("Accidents on the day of a week", fontsize=30)
plt.grid(False)
plt.ylabel('Accident count', fontsize=20)
plt.xlabel('0 - Sunday, 1-Monday, 2 - Tuesday, 3 - Wednesday, 4 - Thursday, 5 - Friday, 6 - Saturday', fontsize=13)

# plot for accidents on the hours of a day
plt.figure(figsize=(12, 6))
accidents.Date_time.dt.hour.hist(rwidth=0.55, alpha=0.50, color='red')
plt.title("Time of the day/night", fontsize=30)
plt.grid(False)
plt.xlabel('Time 0-23 hours', fontsize=20)
plt.ylabel('Accident count', fontsize=10)

# plot for accidents of different age bands
objects = ['0', '0-5', '6-10', '11-15', '16-20', '21-25', '26-35', '36-45', '46-55', '56-65', '66-75', '75+']
plt.figure(figsize=(12, 6))
casualties.Age_Band_of_Casualty.hist(bins=11, rwidth=0.90, alpha=0.50, color='red')
plt.title("Age of people", fontsize=30)
plt.grid(False)
y_pos = np.arange(len(objects))
plt.xticks(y_pos, objects)
plt.xlabel('Age of Drivers', fontsize=20)
plt.ylabel('Accident count', fontsize=20)

#speed zone accidents
speed_zone_accidents = accidents.loc[accidents['Speed_limit'].isin(['20' ,'30' ,'40' ,'50' ,'60' ,'70'])]
speed  = speed_zone_accidents.Speed_limit.value_counts()
explode = (0.0, 0.0, 0.0 , 0.0 ,0.0,0.0) 
plt.figure(figsize=(10,8))
plt.pie(speed.values,  labels=None, 
         autopct='%.1f',pctdistance=0.8, labeldistance=1.9 ,explode = explode, shadow=False, startangle=160,textprops={'fontsize': 15})
plt.axis('equal')
plt.legend(speed.index, bbox_to_anchor=(1,0.7), loc="center right", fontsize=15, 
            bbox_transform=plt.gcf().transFigure)
plt.figtext(.5,.9,'Accidents percentage in Speed Zone', fontsize=25, ha='center')
plt.show()


# Required Datasets
accident_ml = accidents.drop('Accident_Severity', axis=1)
accident_ml = accident_ml[['Age_of_Driver', 'Vehicle_Type', 'Engine_Capacity_(CC)', 'Day_of_Week', 'Weather_Conditions',
                           'Road_Surface_Conditions', 'Age_of_Vehicle', 'Light_Conditions', 'Sex_of_Driver',
                           'Speed_limit']]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(accident_ml.values, accidents['Accident_Severity'].values,
                                                    test_size=0.20, random_state=99)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)


# # Training the Logistic Regression model on the Training set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state=0)
# classifier.fit(X_train, y_train)

# # Training the Decision Tree Classification model on the Training set
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

# Predicting a new result
print(classifier.predict([[20, 5, 599, 1, 1, 1, 9, 4, 1, 30]]))

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)



# saving the model
# joblib.dump(classifier, "mlp.sav")
