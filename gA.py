import pandas as pd
#step 1
sw=pd.read_csv('sparklingwine.csv') 


#step2
sw["good_wine"]=sw["quality"]>5 
print(sw.head())


#step3
training_data=sw.iloc[:900]
validation_data=sw.iloc[900:1200]
testing_data=sw.iloc[1200:]

print(len(training_data))
print(len(validation_data))
print(len(testing_data))

#step4
mu=training_data.mean()
sigma=training_data.mean()

