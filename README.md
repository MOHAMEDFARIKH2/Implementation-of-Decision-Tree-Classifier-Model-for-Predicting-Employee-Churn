# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: H MOHAMED FARIKH
RegisterNumber: 212223080032
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
X=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
X.head()
Y=data["left"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,Y_train)
Y_pred=dt.predict(X_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
data.head()

![image](https://github.com/MOHAMEDFARIKH2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/168570140/27b2bcb2-da89-4efb-bae4-b67be863b902)



data.info()

![image](https://github.com/MOHAMEDFARIKH2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/168570140/75ba62cc-2dd6-4909-bafb-b3f4897f1f1c)



isnull() and sum()

![image](https://github.com/MOHAMEDFARIKH2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/168570140/4d2c5301-d36d-4643-9590-f7391c585a0f)



data value counts()

![image](https://github.com/MOHAMEDFARIKH2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/168570140/5e1787f9-b035-499e-a6f5-e19b38a558b2)


data.head() for salary

![image](https://github.com/MOHAMEDFARIKH2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/168570140/498dd35b-d410-4813-a5c1-cd2e1816559a)



x.head()

![image](https://github.com/MOHAMEDFARIKH2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/168570140/7c3b2727-ffa1-495a-bfbf-ce80f71ee2fd)



accuracy value()

![image](https://github.com/MOHAMEDFARIKH2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/168570140/d268f33e-fcf0-4b9f-a09a-0448a0a63856)



data prediction

![image](https://github.com/MOHAMEDFARIKH2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/168570140/3cbc7e0a-2879-4cb9-b634-c67fb6b9d260)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
