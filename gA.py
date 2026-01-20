import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#step 1
sw=pd.read_csv('sparklingwine.csv') 


#step2
sw["good_wine"]=sw["quality"]>5 
print(sw.head())


#step3
training_data=sw.iloc[:900]
validation_data=sw.iloc[900:1200]
testing_data=sw.iloc[1200:]

print("\nFirst split : 900 training, 300 validation, 400 test")
print(len(training_data))
print(len(validation_data))
print(len(testing_data))

#step4
feature_cols = [col for col in sw.columns if col not in ['quality', 'good_wine']]
mu = training_data[feature_cols].mean()
sigma = training_data[feature_cols].std()
sigma = sigma.replace(0, 1)

xtraining = training_data[feature_cols]
ytraining = training_data['good_wine']
xvalidation = validation_data[feature_cols]
yvalidation = validation_data['good_wine']
xtest = testing_data[feature_cols]
ytest = testing_data['good_wine']
xtraningnorm = (xtraining - mu) / sigma
xvalnorm = (xvalidation - mu) / sigma
xtestnorm = (xtest - mu) / sigma

#step 5 + 6
k_values = range(1, 101)
accuracy = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtraningnorm, ytraining)
    yvalpred = knn.predict(xvalnorm)
    acc = accuracy_score(yvalidation, yvalpred)
    accuracy.append(acc)
    if k % 10 == 0 or k == 1:
        print(f"k={k}:Validation Accuracy={acc:.4f}")


bestk = k_values[np.argmax(accuracy)]
bestacc = max(accuracy)
print(f"\nThe best K is {bestk} with a validation accuracy of {bestacc:.4f}")

#step7
bestknn = KNeighborsClassifier(n_neighbors=bestk)
bestknn.fit(xtraningnorm, ytraining)
ytestprediction = bestknn.predict(xtestnorm)
testacc = accuracy_score(ytest, ytestprediction)
generalizationerror = 1-testacc

print(f"The test Accuracy is {testacc:.4f}")
print(f"And the generalization Error is {generalizationerror:.4f}")

#step8
training_data2 = sw.iloc[:400]
validation_data2 = sw.iloc[400:800]
testing_data2 = sw.iloc[800:1600] 

print("\nSecond split : 400 training, 400 validation, 800 test")
print(len(training_data2))
print(len(validation_data2))
print(len(testing_data2))

feature_cols = [col for col in sw.columns if col not in ['quality', 'good_wine']]
mu2 = training_data2[feature_cols].mean()
sigma2 = training_data2[feature_cols].std()
sigma2 = sigma2.replace(0, 1) 

xtraining2 = training_data2[feature_cols]
ytraining2 = training_data2['good_wine']
xvalidation2 = validation_data2[feature_cols]
yvalidation2 = validation_data2['good_wine']
xtest2 = testing_data2[feature_cols]
ytest2 = testing_data2['good_wine']

xtraningnorm2 = (xtraining2 - mu2) / sigma2
xvalnorm2 = (xvalidation2 - mu2) / sigma2
xtestnorm2 = (xtest2 - mu2) / sigma2

k_values2 = range(1, 101)
accuracy2 = []

for k in k_values2:
    knn2 = KNeighborsClassifier(n_neighbors=k)
    knn2.fit(xtraningnorm2, ytraining2)
    yvalpred2 = knn2.predict(xvalnorm2) 
    acc2 = accuracy_score(yvalidation2, yvalpred2)
    accuracy2.append(acc2)
    if k % 10 == 0 or k == 1:
        print(f"k={k}: Validation Accuracy={acc2:.4f}")

bestk2 = k_values2[np.argmax(accuracy2)]
bestacc2 = max(accuracy2)
print(f"\nThe best K is {bestk2} with a validation accuracy of {bestacc2:.4f}")

bestknn2 = KNeighborsClassifier(n_neighbors=bestk2)
bestknn2.fit(xtraningnorm2, ytraining2)
ytestprediction2 = bestknn2.predict(xtestnorm2)
testacc2 = accuracy_score(ytest2, ytestprediction2)
generalizationerror2 = 1 - testacc2

print(f"The test Accuracy is {testacc2:.4f}")
print(f"And the generalization Error is {generalizationerror2:.4f}")