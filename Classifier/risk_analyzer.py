""" Author: Andrew Dennis"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC 
from sklearn import svm, linear_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from group_data import data, group_main

"""Process data for classification """
df_sent = data()
earn_df = pd.read_csv('D:\\MS Data Science Files\\Thesis\\earnings_data.csv')



df= group_main(df_sent, earn_df)

print(len(df))


# df.to_csv('D:\\MS Data Science Files\\Thesis\\final_df.csv')



X = df[[
		'sum_neg', 
		'sum_unc', 
		'sum_pos', 
		'mean_neg',
		'mean_unc',
		'mean_pos',
		'lag_days',
		'eps_sur',
		'sales_sur'
		]]
y = df[['risk']]


"""Split data for testing algorithms""" 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle= True)  


"""Logistic Classification"""
logist = LogisticRegression()
logist.fit(X_train, y_train)
y_pred_logist = logist.predict(X_test)

"""Logistic Results"""
print(confusion_matrix(y_test,y_pred_logist.round()))
print(' ')  
print('Logistic Regression Results  ')
print(classification_report(y_test,y_pred_logist.round()))  
print('Accuracy %:' , accuracy_score(y_test, y_pred_logist.round())*100) 



"""Random Forrest"""
rf = RandomForestClassifier(n_estimators=200, random_state=42)  
rf.fit(X_train, y_train)  
y_pred = rf.predict(X_test) 


"""Random Forrest Results"""
print(confusion_matrix(y_test,y_pred.round()))
print(' ')  
print('Results Random Forest   ')
print(classification_report(y_test,y_pred.round()))  
print('Accuracy %:' , accuracy_score(y_test, y_pred.round())*100)  


"""Forest Features"""
importances = dict(zip(X_train.columns, rf.feature_importances_))

imp_df = pd.DataFrame(importances.items(), columns=['variable','importance'])

df_importances = imp_df.sort_values(by=['importance'], ascending=False)
print(df_importances)
df_importances['importance'] = df_importances['importance'].astype(float)

imp= sns.barplot(x= 'variable',y= 'importance', data= df_importances, palette= 'Blues_d')
imp.set_xticklabels(imp.get_xticklabels(), rotation= 30)
plt.title("Random Forest Important Variables")
plt.show()


"""Support Vector Machine"""
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(X_train, y_train)  
y_pred_rbf = rbf_svc.predict(X_test) 

"""SVM Results"""
print(confusion_matrix(y_test,y_pred_rbf.round()))
print(' ')  
print('Results SVM  ')
print(classification_report(y_test,y_pred_rbf.round()))  
print('Accuracy %:' , accuracy_score(y_test, y_pred_rbf.round())*100) 


"""Ridge Classification"""
ridge = RidgeClassifier(solver='auto')
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

"""Ridge Results"""
print(confusion_matrix(y_test,y_pred_ridge.round()))
print(' ')  
print('Results Ridge  ')
print(classification_report(y_test,y_pred_ridge.round()))  
print('Accuracy %:' , accuracy_score(y_test, y_pred_ridge.round())*100) 



"""Visualization of Data"""
fig = plt.figure()
ax = fig.add_subplot(111, projection= '3d')

X = df['sales_sur'].astype(float)
Y = df['eps_sur'].astype(float)
Z = df['mean_pos'].astype(float)

ax.set_ylim3d(-100,200)

ax.scatter(X,Y, Z, c= 'r', marker= '+')
ax.set_xlabel('Sales Beat')
ax.set_ylabel('EPS Beat')
ax.set_zlabel('Mean Positivity')
plt.title("3D Scatter of Top 3 Importances")
plt.show()


"""Calculate the fpr and tpr for all thresholds of the classification."""
probs = rf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)


"""ROC CURVE Plot"""
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)
plt.legend(loc= 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

