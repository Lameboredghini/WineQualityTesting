import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
import numpy as np
from sklearn import metrics
from io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
import os
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

#CHANGE ENV PATH
os.environ['PATH'] = os.environ['PATH']+';'+r"C:\Program Files\Graphviz\bin"

#IGNORE WARNINGS
warnings.filterwarnings('ignore')

#READ DATA
df = pd.read_csv('winequality-red.csv')
# print(df.head())
# print(df.describe())
# print(df.shape)

#MISSING DATA
# msno.bar(df)
# plt.show()

#VISUALIZE DATA
# sns.distplot(df['fixed acidity'], color = (0, 0.5, 1), bins = 40, kde = True)
# plt.show()
# df.hist(bins=50, figsize=(20,15)) 
# plt.show()
# sns.countplot(x='quality', data=df)
# plt.show()

#NORMALIZE DATA
scaler = StandardScaler()
# print(df.columns)
df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']] = scaler.fit_transform(df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']])
# print(df.head())
# print(df.describe())

#DIVIDE QUALITY INTO GOOD/BAD
df['quality'] = (df['quality'] >= 7).astype(int)
# sns.countplot(x='quality', data=df)
# plt.show()

#TRAIN TEST SPLIT
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']].values
Y = df['quality'].values
# print(X,Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
# print(x_train.shape, x_test.shape)

#OUTPUT FILE
output = pd.DataFrame()
output[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']] = x_test

#GENERAL MODEL
def general_model(name, model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # print(name, ' : %.2f' %(model.score(x_test, y_test)*100), '%')
    print(name, ' f1 score : %.2f' %(f1_score(y_test, y_pred)*100), '%')
    # df[name] = df[name].fillna()
    # df.to_csv('Output.csv')
    # output[name] = y_pred
    # print(output)
    # output.to_csv('output.csv')

#CONFUSION MATRIX
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#KNN
general_model('KNN', KNeighborsClassifier(n_neighbors=2))
# Ks = 10
# mean_acc = np.zeros((Ks-1))
# std_acc = np.zeros((Ks-1))
# for n in range(1,Ks):
#     neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train, y_train)
#     yhat=neigh.predict(x_test)
#     mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
#     std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
# plt.plot(range(1,Ks),mean_acc,'g')
# plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
# plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
# plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
# plt.ylabel('Accuracy ')
# plt.xlabel('Number of Neighbors (K)')
# plt.tight_layout()
# plt.show()
# knn = KNeighborsClassifier(n_neighbors=2).fit(x_train, y_train)
# ypred = knn.predict(x_test)
# cnf_matrix = confusion_matrix(y_test, ypred, labels=[1,0])
# np.set_printoptions(precision=2)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['quality=1','quality=0'], normalize= False,  title='Confusion matrix')
# plt.show()

#DECISION TREE
general_model('Decision Tree', DecisionTreeClassifier())
# wine_tree = DecisionTreeClassifier(criterion="entropy")
# wine_tree.fit(x_train, y_train)
# ypred = wine_tree.predict(x_test)
# dot_data = StringIO()
# filename = "tree.png"
# featureNames = df.columns[0:11]
# # print(featureNames)
# out=tree.export_graphviz(wine_tree, feature_names=featureNames, out_file=dot_data, class_names= str(np.unique(y_train)), filled=True,  special_characters=True,rotate=False)  
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png(filename)
# img = mpimg.imread(filename)
# plt.figure(figsize=(100, 200))
# plt.imshow(img,interpolation='nearest')
# plt.show()
# dt = DecisionTreeClassifier().fit(x_train, y_train)
# ypred = dt.predict(x_test)
# cnf_matrix = confusion_matrix(y_test, ypred, labels=[1,0])
# np.set_printoptions(precision=2)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['quality=1','quality=0'], normalize= False,  title='Confusion matrix')
# plt.show()

#LOGISTIC REGRESSION
general_model('Logistic Regression', LogisticRegression())
# lr = LogisticRegression().fit(x_train, y_train)
# ypred = lr.predict(x_test)
# cnf_matrix = confusion_matrix(y_test, ypred, labels=[1,0])
# np.set_printoptions(precision=2)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['quality=1','quality=0'], normalize= False,  title='Confusion matrix')
# plt.show()

#SVM
general_model('SVM', SVC(kernel='poly'))
# sv = SVC(kernel='poly').fit(x_train, y_train)
# ypred = sv.predict(x_test)
# cnf_matrix = confusion_matrix(y_test, ypred, labels=[1,0])
# np.set_printoptions(precision=2)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['quality=1','quality=0'], normalize= False,  title='Confusion matrix')
# plt.show()

#XGBOOST
general_model('XGBoost', XGBClassifier(n_jobs = -1, silent = True, verbosity = 0))
# xgb = XGBClassifier(n_jobs = -1, silent = True, verbosity = 0).fit(x_train, y_train)
# ypred = xgb.predict(x_test)
# cnf_matrix = confusion_matrix(y_test, ypred, labels=[1,0])
# np.set_printoptions(precision=2)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['quality=1','quality=0'], normalize= False,  title='Confusion matrix')
# plt.show()