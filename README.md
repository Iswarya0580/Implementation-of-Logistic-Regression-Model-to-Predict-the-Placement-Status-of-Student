# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).
2. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.
3. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.
4. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Iswarya P
RegisterNumber: 212223230082
*/

import pandas as pd
data=pd.read_csv("C:/Users/Manju Mageswari/Downloads/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data["gender"])
data1["ssc_b"]=le.fit_transform(data["ssc_b"])
data1["hsc_b"]=le.fit_transform(data["hsc_b"])
data1["hsc_s"]=le.fit_transform(data["hsc_s"])
data1["degree_t"]=le.fit_transform(data["degree_t"])
data1["workex"]=le.fit_transform(data["workex"])
data1["specialisation"]=le.fit_transform(data["specialisation"])
data1["status"]=le.fit_transform(data["status"])
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#library for large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



```

## Output:
![Screenshot 2024-09-05 103111](https://github.com/user-attachments/assets/ff5ab1da-b0e1-4dec-bc1b-a921b88d412c)


![Screenshot 2024-09-05 103138](https://github.com/user-attachments/assets/5444e1de-0272-4bf9-b924-6c1cbb152ca4)


![Screenshot 2024-09-05 103220](https://github.com/user-attachments/assets/019cd29c-d967-402a-b900-17aa812f2090)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
