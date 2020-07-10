# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:44:27 2020

@author: adars
"""


##########################################################



####### LINEAR REGRESSION 

######  LOGISTIC REGRESSION

##### DECISION TREE

#### RANDOM FOREST

### eXTREME GRADIENT BOOSTING

## K NEAREST NEIGHBORS



###########################################################








# import exploration files 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

file_path = 'path_to_data.csv'

# read in data 
data = pd.read_csv(file_path)


############################################################################## 
#Data Exploration
##############################################################################

#rows and columns returns (rows, columns)
data.shape

#returns the first x number of rows when head(num). Without a number it returns 5
data.head()

#returns the last x number of rows when tail(num). Without a number it returns 5
data.tail()

#returns an object with all of the column headers 
data.columns

#basic information on all columns 
data.info()

#gives basic statistics on numeric columns
data.describe()

#shows what type the data was read in as (float, int, string, bool, etc.)
data.dtypes

#shows which values are null
data.isnull()

#shows which columns have null values
data.isnull().any()

#shows for each column the percentage of null values 
data.isnull().sum() / data.shape[0]

#plot histograms for all numeric columns 
data.hist() 


############################################################################## 
#Data Manipulation
##############################################################################

# rename columns 
data.rename(index=str columns={'col_oldname':'col_newname'})

# view all rows for one column
data.col_name 
data['col_name']

# multiple columns by name
data[['col1','col2']]
data.loc[:['col1','col2']]

#columns by index 
data.iloc[:,[0:2]]

# drop columns 
data.drop('colname', axis =1) #add inplace = True to do save over current dataframe
#drop multiple 
data.drop(['col1','col2'], axis =1)

#lambda function 
data.apply(lambda x: x.colname**2, axis =1)

# pivot table 
pd.pivot_table(data, index = 'col_name', values = 'col2', columns = 'col3')

# merge  == JOIN in SQL
pd.merge(data1, data2, how = 'inner' , on = 'col1')

# write to csv 
data.to_csv('data_out.csv')







##############################################################


##################  LINEAR REGRESSION    #####################


###############################################################


# split the dataset into train and test
# --------------------------------------
train, test = train_test_split(conc, test_size = 0.3)
print(train.shape)
print(test.shape)

# split the train and test into X and Y variables
# ------------------------------------------------
train_x = train.iloc[:,0:8]; train_y = train.iloc[:,8]
test_x  = test.iloc[:,0:8];  test_y = test.iloc[:,8]

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

train_x.head()
train_y.head()

train.head()

# ensure that the X variables are all numeric for regression
# ----------------------------------------------------------
train.dtypes
    

# To add the constant term A (Y = A + B1X1 + B2X2 + ... + BnXn)
# Xn = ccomp,slag,flyash.....
# ----------------------------------------------------------
lm1 = sm.OLS(train_y, train_x).fit()

# Prediction
# -----------------
pdct1 = lm1.predict(test_x)
print(pdct1)


# store the actual and predicted values in a dataframe for comparison
# -------------------------------------------------------------------
actual = list(test_y.head(50))
type(actual)
type(predicted)
predicted = np.round(np.array(list(pdct1.head(50))),2)
print(predicted)

# Actual vs Predicted
#-----------------------
df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})
print(df_results)


#To Check the Accuracy:
#-----------------------------
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1)))  





#######################





##############################################################


##################  LOGISTIC REGRESSION    #####################


###############################################################



#Split dataset in features and target variable
#-----------------------------------------------
total_cols = len(diab.columns)
total_cols
X=diab.values[:,0:total_cols-1]
Y=diab.values[:,total_cols-1]


# Split X and y into training and testing sets
#------------------------------------------------
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# Instantiate the model (using the default parameters)
#-------------------------------------------------------
logreg = LogisticRegression()

# Fit the model with data
#------------------------------
logreg.fit(X_train,y_train)

# Prediction of the model
#----------------------------
y_pred=logreg.predict(X_test)
y_pred


# Predict on the test set
# ---------------------------
pred_y = logreg.predict(X_test)
labels=[0,1]
cf = confusion_matrix(y_pred,y_test,labels)
print(cf)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Confusion matrix with details

#Class 1 : Positive
#Class 2 : Negative
#Definition of the Terms:

#Positive (P) : Observation is positive (for example: is an apple).
#Negative (N) : Observation is not positive (for example: is not an apple).
#True Positive (TP) : Observation is positive, and is predicted to be positive.
#False Negative (FN) : Observation is positive, but is predicted negative.
#True Negative (TN) : Observation is negative, and is predicted to be negative.
#False Positive (FP) : Observation is negative, but is predicted positive.
#Classification Rate/Accuracy:
#Classification Rate or Accuracy is given by the relation:

# -----------------------------
ty=list(y_test)
print(ty)
py=list(pred_y)
print(py)
cm1=ConfusionMatrix(py,ty)
print(cm1)
cm1.print_stats()
cm1.plot()


# Classification report : precision, recall, F-score
# ---------------------------------------------------
print(cr(y_test, pred_y))



#####################################

##############################################################


##################  DECISION TREE    #####################


###############################################################



total_cols = len(con.columns)
total_cols
x = con.values[:,:total_cols-1]
y = con.values[:,total_cols-1]


#split the data into training set and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state = 0)
print(x_train)
print(y_train)
print(x_test)
print(y_test)


# import the regressor 
from sklearn.tree import DecisionTreeRegressor

# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 0)
  
# fit the regressor with X and Y data 
regressor.fit(x_train, y_train) 


#Predicting labels on the test set.
y_pred =  regressor.predict(x_test)
print(y_pred)






#################################



##############################################################


##################  RANDOM FOREST    #####################


###############################################################


# Fitting Random Forest Regression to the dataset 
# import the regressor 
from sklearn.ensemble import RandomForestRegressor 

# create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 

# fit the regressor with x and y data 
regressor.fit(x, y) 



Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1)) # test the output by changing values 


# Visualising the Random Forest Regression results 

# arange for creating a range of values 
# from min value of x to max 
# value of x with a difference of 0.01 
# between two consecutive values 
X_grid = np.arange(min(x), max(x), 0.01) 

# reshape for reshaping the data into a len(X_grid)*1 array, 
# i.e. to make a column out of the X_grid value				 
X_grid = X_grid.reshape((len(X_grid), 1)) 

# Scatter plot for original data 
plt.scatter(x, y, color = 'blue') 

# plot predicted data 
plt.plot(X_grid, regressor.predict(X_grid), 
		color = 'green') 
plt.title('Random Forest Regression') 
plt.xlabel('Position level') 
plt.ylabel('Salary') 
plt.show()





#############################


##############################################################


##################  eXTREME GRADIENT BOOSTING    #####################


###############################################################



# Write Python3 code here 
# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset 
dataset = pd.read_csv('Churn_Modelling.csv') 
X = dataset.iloc[:, 3:13].values 
y = dataset.iloc[:, 13].values 

# Encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_X_1 = LabelEncoder() 

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) 
labelencoder_X_2 = LabelEncoder() 

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) 
onehotencoder = OneHotEncoder(categorical_features = [1]) 

X = onehotencoder.fit_transform(X).toarray() 
X = X[:, 1:] 

# Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( 
		X, y, test_size = 0.2, random_state = 0) 

# Fitting XGBoost to the training data 
import xgboost as xgb 
my_model = xgb.XGBClassifier() 
my_model.fit(X_train, y_train) 

# Predicting the Test set results 
y_pred = my_model.predict(X_test) 

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 




###########################

##############################################################


##################  K NEAREST NEIGHBORS    #####################


###############################################################






from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler() 

scaler.fit(df.drop('TARGET CLASS', axis = 1)) 
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1)) 

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1]) 
df_feat.head() 



from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( 
	scaled_features, df['TARGET CLASS'], test_size = 0.30) 

# Remember that we are trying to come up 
# with a model to predict whether 
# someone will TARGET CLASS or not. 
# We'll start with k = 1. 

from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 1) 

knn.fit(X_train, y_train) 
pred = knn.predict(X_test) 

# Predictions and Evaluations 
# Let's evaluate our KNN model ! 
from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(y_test, pred)) 

print(classification_report(y_test, pred)) 






error_rate = [] 

# Will take some time 
for i in range(1, 40): 
	
	knn = KNeighborsClassifier(n_neighbors = i) 
	knn.fit(X_train, y_train) 
	pred_i = knn.predict(X_test) 
	error_rate.append(np.mean(pred_i != y_test)) 

plt.figure(figsize =(10, 6)) 
plt.plot(range(1, 40), error_rate, color ='blue', 
				linestyle ='dashed', marker ='o', 
		markerfacecolor ='red', markersize = 10) 

plt.title('Error Rate vs. K Value') 
plt.xlabel('K') 
plt.ylabel('Error Rate') 



# FIRST A QUICK COMPARISON TO OUR ORIGINAL K = 1 
knn = KNeighborsClassifier(n_neighbors = 1) 

knn.fit(X_train, y_train) 
pred = knn.predict(X_test) 

print('WITH K = 1') 
print('\n') 
print(confusion_matrix(y_test, pred)) 
print('\n') 
print(classification_report(y_test, pred)) 


# NOW WITH K = 15 
knn = KNeighborsClassifier(n_neighbors = 15) 

knn.fit(X_train, y_train) 
pred = knn.predict(X_test) 

print('WITH K = 15') 
print('\n') 
print(confusion_matrix(y_test, pred)) 
print('\n') 
print(classification_report(y_test, pred)) 







































