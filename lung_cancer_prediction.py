#1#
import numpy as np
import pandas as pd
#2#
lung=pd.read_csv('/content/survey lung cancer.csv')
#3#
lung #calling#
#4#
lung.shape
#5#
lung.info
#6#
lung.columns
#output#
#Index(['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
       'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
       'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
       'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER'],
      dtype='object')#
#7#
lung.isnull().sum()
#8#
lung=lung.dropna()#eliminates the null#
#9#
lung.isnull().sum()
#10# #converting string to binary or float of input#
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
lung['GENDER']=label_encoder.fit_transform(lung['GENDER'])
lung['GENDER'].unique()
#11#
#calling x to verify#
x
#converting string to binary or float of output#
#12#
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
lung['LUNG_CANCER']=label_encoder.fit_transform(lung['LUNG_CANCER'])
lung['LUNG_CANCER'].unique()
#13#
y=lung[['LUNG_CANCER']]
#calling to verify#
#14#
y
#training and testing##15#
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
#16#
x_train.shape
#17#
x_test.shape
#18#
y_train.shape
#19#
y_test.shape
#classification algorithm#
   #svm#
   #gradientboost#
   #knn#
   #decision tree#
   #random forest#
   #gaussian nb#
   #adaboost#
   #xgb#
   #cataboost#
#20#svm#
from sklearn.metrics import accuracy_score
from sklearn import svm
cif=svm.SVC()
cif.fit(x,y)
y_pred=cif.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
#21#decision tree#
from sklearn.metrics import accuracy_score
from sklearn import tree
c = tree.DecisionTreeClassifier()
c.fit(x,y)
y_pred=c.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
#22#randomforest#
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
a = RandomForestClassifier(n_estimators=10)
a.fit(x,y)
y_pred=a.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
#23#knn#
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
n = KNeighborsClassifier(n_neighbors=3)
n.fit(x,y)
y_pred=n.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
#24#guassiannb#
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
g = GaussianNB()
g.fit(x,y)
y_pred=g.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
#25#gradientboost#
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
s = GradientBoostingClassifier(n_estimators=100)
s.fit(x,y)
y_pred=s.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
#26#adaboost#
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, algorithm="SAMME",)
clf.fit(x,y)
y_pred=clf.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
#27#xgb#
from sklearn.metrics import accuracy_score
import xgboost as xgb
b=xgb.XGBClassifier()
b.fit(x,y)
y_pred=b.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
#28#cataboost install#
!pip install catboost
#29#cataboost#
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
m=CatBoostClassifier(iterations=100,learning_rate=0.1,depth=6)
m.fit(x,y)
y_pred=m.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
#prediction#
p=clf.predict([[1,69,1,2,2,1,1,2,1,2,2,2,2,2,2]])
#output#
print(p)
#piechart#
import matplotlib.pyplot as plt
label=['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY']
values=[100,691,111,222,234]
color=['#CD5C5C','#F08080','#FA8072','#E9967A','#FFA07A']
plt.figure(figsize=(10,6))
plt.pie(values,labels=label,colors=color,autopct='%1.1f%%',startangle=160)
plt.axis('equal')
plt.title('pie chart')
plt.show()
#barchart#
import matplotlib.pyplot as plt
categories=['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY']
values=[100,691,111,222,234]
color=['#CD5C5C','#F08080','#FA8072','#E9967A','#FFA07A']
plt.figure(figsize=(10,6))
plt.bar(categories,values,color=color)
plt.xlabel('categories')
plt.title('sample bar chart')
plt.show()
#end of classification algorithm#


