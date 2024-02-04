import copy
import numpy as np
import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
acc_log=[]

# ----------------------ABOUT DATASET----------------------
'''
1 - read data
2 - print first five rows
3 - print data info : range index - how many entries, how many cols, show all cols - non-null, count and datatype,
    summarize data types of cols, how many memory it use
4 - print how many rows for each col is null, how many values per cols is missing

sex = 0 - femlae, sex = 1 - male
'''
data = pd.read_csv('projekat/heart.csv')
print(data.head())
data.info()
print(data.isnull().sum())

'''
Analyze each column
    Count: The number of non-null values for each column
    Mean : The average value of the data points in each column
    Standard Deviation (std): A measure of the amount of variation or dispersion of a set of values.
        It quantifies how much the values in a column deviate from the mean.
        A higher standard deviation indicates greater variability.
    Minimum: The smallest value in each column
    25th Percentile (Q1 or First Quartile): Also known as the lower quartile, this is the value below which 25% of the data falls.
        It is the point at which the data set is divided into the lowest 25% of values.
    Median (50th Percentile or Q2): The middle value of the data set when it is sorted in ascending order.
        It separates the higher half from the lower half. Also known as the second quartile.
    75th Percentile (Q3 or Third Quartile): The value below which 75% of the data falls. 
        It is the point at which the data set is divided into the lowest 75% of values. Also known as the upper quartile.
    Maximum: The largest value in each column
'''
print(data.describe().T)


'''
COUNT HOW MANY MALES AND FEMALES WHERE ANALYZED
value_counts() is used to count the occurrences of each category in the 'sex' column.
plt.bar() is used to create a bar chart, where x.index represents the categories ('male' and 'female') and x represents the counts.
The colors are set using the color parameter in plt.bar().
Finally, the plot is displayed using plt.show().

People having gender as 0 are more than twice the people having gender as 1
'''
# x=(data.sex.value_counts())
# print(f'Number of males is {x[1]} and number of females is {x[0]}')
# plt.bar(['Male', 'Female'], x, color=['blue', 'orange']) 
# plt.title('Countplot of Sex')
# plt.xlabel('Sex')
# plt.ylabel('Count')
# plt.show()

'''
COUNT CHEST PAIN TYPES
Value 0: typical angina - This type of chest pain is usually associated with coronary artery disease.
    It is characterized by a specific pattern of discomfort or pain that occurs when the data muscle does not receive enough blood and oxygen.
    The pain is typically described as a squeezing or pressure-like sensation in the chest and may radiate to the left arm, neck, jaw, shoulder, or back.
    It often occurs during physical activity or stress and is relieved by rest or nitroglycerin. 
Value 1: atypical angina - Atypical angina refers to chest pain that doesn't fit the classic pattern of typical angina.
    The discomfort may be less specific, and the pain characteristics may vary. 
Value 2: non-anginal pain - Non-anginal chest pain refers to discomfort in the chest that is not related to the data.
    It may be caused by issues such as musculoskeletal problems, gastrointestinal disorders, or anxiety.
    Unlike angina, non-anginal pain is not associated with a lack of blood flow to the data muscle.
Value 3: asymptomatic - Asymptomatic means the absence of symptoms. In the context of chest pain and cardiac conditions,
    it means that the individual is not experiencing any noticeable chest pain or discomfort.
    However, this doesn't necessarily mean the absence of underlying data issues; it simply means that the person is not currently feeling any chest pain.

It can be observed people have chest pain of type 0 i.e 'Typical Angina' is the highest.
It can be observed people have chest pain of type 3 i.e 'Asymptomatic' is the lowest
It can also be observed people with chest pain of type 0 is almost 50% of all the people.    
    
'''
# x = data['cp'].value_counts()
# print(x)
# plt.bar(x.index, x, color=['blue', 'orange', 'green', 'red'])  
# plt.xticks(x.index, ['typical angine','atypical-angine', 'non-anginal', 'asymptomatic'])
# plt.title('Countplot of cp')
# plt.xlabel('Chest Pain Category')
# plt.ylabel('Count')
# plt.show()


'''
FAST BLOOD SUGAR
if fbs is greater than 120 mg/dl then value in tha row is 1, otherwise its 0
'''
# x = data['fbs'].value_counts()
# print(x)
# plt.bar(x.index, x, color=['blue', 'orange'])  
# plt.xticks(x.index, ['low','high'])
# plt.title('Countplot of fbs')
# plt.xlabel('Blood Sugar Level')
# plt.ylabel('Count')
# plt.show()


'''
RESTING ELECTROCARDIOGRAPHY RESULTS 
Value 0: normal
Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    Abnormalities in these regions can be indicative of various cardiac conditions, 
    including ischemia (inadequate blood flow to the heart muscle) or infarction (heart attack). 
Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
    Left ventricular hypertrophy (LVH) is a condition where the walls of the left ventricle of the heart are thicker than normal. 

ECG count is almost the same for type 0 and 1. Also, for type 2 its almost negligible in comparision to type 0 and 1.
'''
# x = data['restecg'].value_counts()
# print(x)
# plt.bar(x.index, x, color=['blue', 'orange', 'green'])  
# plt.xticks(x.index, ['ST_T','normal', 'LVH'])
# plt.title('Countplot of rest_ecg')
# plt.xlabel('Resting ECG Results')
# plt.ylabel('Count')
# plt.show()
'''
EXERCISE INDUCED ANGINA
VALUE - 1 = yes; 0 = no

exercise induced angina - chest pain or discomfort that occurs during physical activity or exertion.
    It is a symptom commonly associated with coronary artery disease (CAD),
    a condition where the blood vessels supplying the heart muscle (coronary arteries) become narrowed or blocked, reducing blood flow to the heart.
    During exercise, the heart's demand for oxygen-rich blood increases. If the coronary arteries are partially blocked by atherosclerosis (buildup of plaque),
    they may not be able to deliver enough blood to meet the increased demand. 
    This imbalance between oxygen supply and demand can lead to chest pain or discomfort, known as angina pectoris.
'''
# x = data['exng'].value_counts()
# print(x)
# plt.bar(x.index, x, color=['blue', 'orange'])  
# plt.xticks(x.index, ['no','yes'])
# plt.title('Countplot of exang')
# plt.xlabel('Exercise induced angina')
# plt.ylabel('Count')
# plt.show()
'''
COUNT TARGET
VALUE - 0= less chance of heart attack 1= more chance of heart attack
'''
# x = data['output'].value_counts()
# print(x)
# plt.bar(x.index, x, color=['blue', 'orange'])  
# plt.xticks(x.index, ['yes','no'])
# plt.title('Countplot of target')
# plt.xlabel('Chances of heart attack')
# plt.ylabel('Count')
# plt.show()

'''
remove duplicates for improving the data quality and accuracy of machine learning models
only 1 duplicates are there, so we dont have big data loss and we can remove it
'''
# print(data[data.duplicated()])
# print(data.drop_duplicates(keep='first',inplace=True))
# print('Number of rows are',data.shape[0], 'and number of columns are ',data.shape[1])
'''
CORRELATION MATRIX
ANALIZA ??????????????????????????
'''
# plt.figure(figsize=(15,10)) # da podesimo velicinu grafika

# correlation_matrix = data.corr() # racunamo matricu korelacije

# sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.1)
# plt.title('Correlation matrix')
# plt.show()
'''
AGE DENSITY DISTRIBUTION
Density distribution is highest for age group 55 to 60
'''
# plt.figure(figsize=(10, 10))
# plt.hist(data['age'], bins=30, color='blue', label='Age', density=True)
# plt.title('Histogram of Age with KDE')
# plt.xlabel('Age')
# plt.ylabel('Density')
# sb.histplot(data['age'], kde=True, kde_kws=dict(cut=3), color='blue')
# plt.legend()
# plt.show()

'''
RESTING BLOOD PRESSURE DENSITY DISTRIBUTION
resting blood pressure (in mm Hg)
Trtbs has the highest count around 130
'''
# plt.figure(figsize=(5,4.5))
# plt.hist(data['trtbps'], bins=30, color='blue', label='Resting Blood Pressure', density=True)
# plt.title('Histogram of Resting Blood Pressure with KDE')
# plt.xlabel('Resting Blood Pressure')
# plt.ylabel('Density')
# sb.histplot(data['trtbps'], kde=True, kde_kws=dict(cut=3), color='blue')
# plt.legend()
# plt.show()


'''
THALIUM STRESS TEST RESULTS DISTRIBUTION
VALUE - maximum heart rate achieved
Thalium Stress Test - a diagnostic procedure used to assess the blood flow to the hrat muscle.
    It combines exercise stress (usually on a treadmill or stationary bicycle) with the administration of a radioactive substance (thallium or technetium)
    and imaging to evaluate the circulation in the coronary arteries.
thal has the highest couachhnt around 160

'''

# plt.figure(figsize=(5,4.5))
# plt.hist(data['thalachh'], bins=30, color='blue', label='Thalium stress test results', density=True)
# plt.title('Histogram of Thalium stress test results with KDE')
# plt.xlabel('Thalium stress test results')
# plt.ylabel('Density')
# sb.histplot(data['thalachh'], kde=True, kde_kws=dict(cut=3), color='blue')
# plt.legend()
# plt.show()

'''
Display of distributions of all columns
'''
# plt.figure(figsize=(15,10))
# for i,col in enumerate(data.columns,1):
#     plt.subplot(5,3,i)
#     plt.title(f"Distribution of {col} Data")
#     sb.histplot(data[col],kde=True)
#     plt.tight_layout()
#     plt.plot()
# plt.show()
"""
ATTACK VERSUS AGE
"""

# plt.figure(figsize=(10,10))
# sb.distplot(data[data['output'] == 0]["age"], color='green',kde=True,) 
# sb.distplot(data[data['output'] == 1]["age"], color='red',kde=True)
# plt.title('Attack versus Age')
# plt.show()
'''
Shows the Distribution of Heat Diseases with respect to male and female using seaborn

'''
# plt.figure(figsize=(10,10))
# sb.distplot(data[data['output'] == 0]["age"], color='green',kde=True,) 
# sb.distplot(data[data['output'] == 1]["age"], color='red',kde=True)
# plt.title('data attac chances versus Age')
# plt.show()
'''
LOGICAL REGRESSION
'''
df1 = data.copy()

cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]

df1 = pd.get_dummies(df1, columns = cat_cols, drop_first = True)

x = df1.drop(columns=['output'])
y = df1[['output']]

scaler = preprocessing.StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
x_train[con_cols] = scaler.fit_transform(x_train[con_cols])
x_test[con_cols] = scaler.transform(x_test[con_cols])


print("The first 5 rows of data set are")
print(x.head())
model_logreg = LogisticRegression()

model_logreg.fit(x_train, y_train)

y_pred_proba = model_logreg.predict_proba(x_test)

y_pred = np.argmax(y_pred_proba,axis=1)


print("The test accuracy score of Logistric Regression is ", accuracy_score(y_test, y_pred))

conf = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix : \n", conf)
print ("The accuracy of Logistic Regression is : ", accuracy_score(y_test, y_pred)*100, "%")


# LOGICAL REGRESSION 2
df2= data.copy()
x = df2.drop(columns=['output'])
y = df2[['output']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

scaler=preprocessing.RobustScaler()
x_train[con_cols] = scaler.fit_transform(x_train[con_cols])
x_test[con_cols] = scaler.transform(x_test[con_cols])


lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred_proba = lr.predict_proba(x_test)

y_pred = np.argmax(y_pred_proba,axis=1)
print("The test accuracy score of Logistric Regression is ", accuracy_score(y_test, y_pred))

conf = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix : \n", conf)
print ("The accuracy of Logistic Regression is : ", accuracy_score(y_test, y_pred)*100, "%")


"""
KNN
"""
df3= data.copy()
x = df3.drop(columns=['output'])
y = df3[['output']]
df3 = pd.get_dummies(df3, columns = cat_cols, drop_first = True)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

scaler=StandardScaler()
x_train[con_cols] = scaler.fit_transform(x_train[con_cols])
x_test[con_cols] = scaler.transform(x_test[con_cols])

error_rate = []
  
for i in range(1, 40):
      
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(x_train, y_train)
    pred_i = model.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test['output']))
  
plt.figure(figsize =(10, 6))
plt.plot(range(1, 40), error_rate, color ='blue',
                linestyle ='dashed', marker ='o',
         markerfacecolor ='red', markersize = 10)
  
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

model = KNeighborsClassifier(n_neighbors = 31)
  
model.fit(x_train, y_train)
predicted = model.predict(x_test)
  
print('Confusion Matrix :')
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predicted))

print()
print()
print("The accuracy of KNN is : ", accuracy_score(y_test, predicted.round())*100, "%")

model = KNeighborsClassifier(n_neighbors = 17)
  
model.fit(x_train, y_train)
predicted = model.predict(x_test)
  
print('Confusion Matrix :')
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predicted))

print()
print()
print("The accuracy of KNN is : ", accuracy_score(y_test, predicted.round())*100, "%")


model = KNeighborsClassifier(n_neighbors = 23)
  
model.fit(x_train, y_train)
predicted = model.predict(x_test)
  
print('Confusion Matrix :')
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predicted))

print()
print()
print("The accuracy of KNN is : ", accuracy_score(y_test, predicted.round())*100, "%")

"""
SVC
"""
df4= data.copy()
df4 = pd.get_dummies(df4, columns = cat_cols, drop_first = True)

x = df4.drop(columns=['output'])
y = df4[['output']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

scaler=StandardScaler()
x_train[con_cols] = scaler.fit_transform(x_train[con_cols])
x_test[con_cols] = scaler.transform(x_test[con_cols])
"""
svc parms test
"""
# df_nontree=data
# target="output"
# y=df_nontree[target].values
# df_nontree.drop("output",axis=1,inplace=True)
# df_nontree=pd.concat([df_nontree,df4[target]],axis=1)
# df_nontree.head()
# feature_col_nontree=df_nontree.columns.to_list()
# feature_col_nontree.remove(target)
# acc_svm_rbf=[]
# kf=model_selection.StratifiedKFold(n_splits=5)
# for fold , (trn_,val_) in enumerate(kf.split(X=df_nontree,y=y)):
    
#     X_train=df_nontree.loc[trn_,feature_col_nontree]
#     y_train=df_nontree.loc[trn_,target]
    
#     X_valid=df_nontree.loc[val_,feature_col_nontree]
#     y_valid=df_nontree.loc[val_,target]
    
#     ro_scaler=StandardScaler()
#     X_train=ro_scaler.fit_transform(X_train)
#     X_valid=ro_scaler.transform(X_valid)
#     clf=SVC(kernel="rbf")
#     clf.fit(X_train,y_train)
#     print(f"Parameters for Fold {fold}: {clf.get_params()}")
#     y_pred=clf.predict(X_valid)
#     print(f"The fold is : {fold} : ")
#     print(classification_report(y_valid,y_pred))
#     acc=roc_auc_score(y_valid,y_pred)
#     acc_svm_rbf.append(acc)
#     print(f"The accuracy for {fold+1} : {acc}")
    
#     pass

# svc1
model = SVC()
model.fit(x_train, y_train)
predicted = model.predict(x_test)
print("The accuracy of SVM is : ", accuracy_score(y_test, predicted)*100, "%")

# SVC 2

clf = SVC(kernel='rbf', C=1, degree=3, gamma='scale', random_state=None).fit(x_train,y_train)

y_pred = clf.predict(x_test)
print("The test accuracy score of SVM is ", accuracy_score(y_test, y_pred))

# # svc 3
df5= data.copy()
df5 = pd.get_dummies(df5, columns = cat_cols, drop_first = True)

x = df5.drop(columns=['output'])
y = df5[['output']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

scaler=StandardScaler()
x_train[con_cols] = scaler.fit_transform(x_train[con_cols])
x_test[con_cols] = scaler.transform(x_test[con_cols])

'''
search the params for svc
'''
# svm = SVC()

# c = [0.01, 0.1, 1.0, 10.0, 90, 100.0, 110]
# gamma = [0.001, 0.001, 0.01, 0.1, 1]
# degree = [2,3,4]
# param_grid=[{'C': c,'kernel': ['linear']},
#             {'C': c,'kernel': ['rbf'],'gamma': gamma}
#         ]

# searcher = GridSearchCV(estimator=svm, 
#                         param_grid=param_grid, 
#                         scoring='f1', 
#                         refit=True, 
#                         n_jobs=-1, 
#                         verbose=4)

# searcher.fit(x_train,y_train)
# print("The best params are :", searcher.best_params_)
# print("The best score is   :", searcher.best_score_)

# y_pred = searcher.predict(x_test)

# print("The test accuracy score of SVM after hyper-parameter tuning is ", accuracy_score(y_test, y_pred))



# SVC 4

clf = SVC(kernel='linear', C=10, gamma = 0.01,random_state=42).fit(x_train,y_train)

y_pred = clf.predict(x_test)

print("The test accuracy score of SVM is ", accuracy_score(y_test, y_pred))






