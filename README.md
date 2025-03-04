# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: K.HEMANATH 
RegisterNumber: 212223100012 
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

# display the content in datafield
df.head()
df.tail()

#Segregating data to variables

x = df.iloc[:,:-1].values
print(x)


y = df.iloc[:,1].values
print(y)

#Splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#displaying predicted values
print(y_pred)

#displaying actual values
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

Head And Tail:

![image](https://github.com/user-attachments/assets/ca21d781-35a1-46ed-af4a-8ffdd54dcc9f)

Segregating Data To Variables:

![image](https://github.com/user-attachments/assets/0f4d142f-ee88-42ac-8246-76e006b74eb5)

Displaying Predicted Values:

![image](https://github.com/user-attachments/assets/d719b6a3-0c6e-45df-be06-9eb729c6489b)

Displaying Actual Values:

![image](https://github.com/user-attachments/assets/e3d08173-a0f5-4116-a41b-ec1ce3614c11)

Graph Plot For Training Data:

![image](https://github.com/user-attachments/assets/315ff33f-ecca-4db6-acff-e111713acf87)

Graph plot for test data:

![image](https://github.com/user-attachments/assets/f5f31e48-f13b-40d6-a655-9bb72e221c84)

MSE MAE RMSE:

![image](https://github.com/user-attachments/assets/f9984a84-381c-4943-aca0-260a03174970)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
