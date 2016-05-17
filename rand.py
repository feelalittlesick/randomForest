print('Importing libraries...')
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("Data/cancerTrain.csv", delimiter=';', )
test  = pd.read_csv("Data/cancerTest.csv", delimiter=';', )
train.info()

print('Defining submission file...')    
def create_submission(rfc, train, test, predictors, filename):
    rfc.fit(train[predictors], train["Class"])
    predictions = rfc.predict(test[predictors])
    submission = pd.DataFrame({
        "PatientId": test["PatientId"],
        "Class": predictions
    })
    submission.to_csv(filename, index=False)

predictors = ["ClumpThickness", "UniformityCellSize", "CellShape", "MarginalAdhesion", "CellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses"]

rfc = RandomForestClassifier(n_estimators=100)
print('Creating submission...')
create_submission(rfc, train, test, predictors, "predictions.csv")
print('Submitted.')