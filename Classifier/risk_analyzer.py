import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from group_data import data, group_main

"""Process data for classification """
df_sent = data()
earn_df = pd.read_csv('D:\\MS Data Science Files\\Thesis\\earnings_data.csv')

df= group_main(df_sent, earn_df)

df.to_csv('D:\\MS Data Science Files\\Thesis\\final_df.csv')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)  


"""Random Forrest"""
regressor = RandomForestClassifier(n_estimators=200, random_state=42)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test) 


"""Random Forrest Results"""
print(confusion_matrix(y_test,y_pred.round()))
print(' ')  
print('Top Three Variables    ')
print(classification_report(y_test,y_pred.round()))  
print('Accuracy %:' , accuracy_score(y_test, y_pred.round())*100)  



"""Forest Features"""
importances = dict(zip(X_train.columns, regressor.feature_importances_))

imp_df = pd.DataFrame(importances.items(), columns=['variable','importance'])


df_importances = imp_df.sort_values(by=['importance'], ascending=False)
df_importances['importance'] = df_importances['importance'].astype(float)

imp= sns.barplot(x='variable',y='importance', data=df_importances, palette='Blues_d')
imp.set_xticklabels(imp.get_xticklabels(), rotation=30)
plt.show()


"""Visualization of Data"""
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

X = df['sales_sur'].astype(float)
Y = df['eps_sur'].astype(float)
Z = df['mean_pos'].astype(float)

ax.set_ylim3d(-100,200)

ax.scatter(X,Y, Z, c='r', marker='+')
ax.set_xlabel('Sales Beat')
ax.set_ylabel('EPS Beat')
ax.set_zlabel('Mean Positivity')
plt.show()



"""Calculate the fpr and tpr for all thresholds of the classification."""
probs = regressor.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

"""ROC CURVE Plot"""
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

