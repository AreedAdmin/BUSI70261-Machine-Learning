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