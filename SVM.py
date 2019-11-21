#------------------------------------------------linear SVM--------------------------------------------------------

import pandas as pd import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix


bankdata = pd.read_csv("D:\\Sahil_py\\data\\bill_authentication.csv") bankdata.shape
bankdata.head()


x=bankdata.drop('Class',axis=1) y=bankdata['Class']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


svclassifier=SVC(kernel='linear') svclassifier.fit(x_train,y_train)


y_pred=svclassifier.predict(x_test)


print(classification_report(y_test,y_pred))


Output:

precision	recall f1-score support


0	1.00	0.99	1.00	151

1	0.99	1.00	1.00	124


accuracy	1.00	275

macro avg	1.00	1.00	1.00	275

weighted avg	1.00	1.00	1.00	275




Algorithm for non-linear SVM:

Step1-Import required python libraries required Step2-import the iris dataset from sklearn Step3-Preprocess the data
Step4-Devide the data into training dataset and testing dataset(input and output).

Step5-Implement the non-linear SVM kernel(polynomial,guasian,sigmoid) on the training dataset.
Step6-Compute the predicted value.

Step7-Compare the predicted value with testing dataset and determine accuracy of the classifier.




#-----------------------------------------------non-Linear SVM----------------------------------------------

import pandas as pd import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix from sklearn import datasets


irisdata = datasets.load_iris() #digits=
print(irisdata)


#irisdata.shape #irisdata.head()


x=irisdata.data y=irisdata.target


print(x) print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


#---------------------------------polynomial kernel-----------------------------------


svclassifier=SVC(kernel='poly',degree=8) svclassifier.fit(x_train,y_train)


y_pred=svclassifier.predict(x_test) print("Output of Polynomial kernel") print(confusion_matrix(y_test,y_pred)) print(classification_report(y_test,y_pred))


#----------------------------------Gausian RBF kernel--------------------------------------


svclassifier=SVC(kernel='rbf') svclassifier.fit(x_train,y_train)


y_pred=svclassifier.predict(x_test) print("Output of Guasian kernel") print(confusion_matrix(y_test,y_pred)) print(classification_report(y_test,y_pred))



#-----------------------------------Sigmoid Kernel-------------------------------------


svclassifier=SVC(kernel='sigmoid') svclassifier.fit(x_train,y_train)


y_pred=svclassifier.predict(x_test) print("Output of Sigmoid kernel") print(confusion_matrix(y_test,y_pred)) print(classification_report(y_test,y_pred))
