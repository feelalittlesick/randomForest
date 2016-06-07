print('Importing libraries...')
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("Data/cancerTrain.csv", delimiter=';', )
test_data  = pd.read_csv("Data/cancerTest.csv", delimiter=';', )
to_check   = pd.read_csv("Data/toCheck.csv", delimiter=';', )

print('Defining submission file...')    
def create_submission(rfc, train, test, to_check, predictors, filename):
    rfc.fit(train[predictors], train["Class"])
    predictions = rfc.predict(test[predictors])
    print(confusion_matrix(predictions, to_check["Class"], labels=[2,3,4]))
    submission = pd.DataFrame({
        "PatientId": test["PatientId"],
        "Class": predictions
    })
    submission.to_csv(filename, index=False)

predictors = ["ClumpThickness", "UniformityCellSize", "CellShape", "MarginalAdhesion", "CellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses"]

print('Finding best n_estimators for RandomForestClassifier...')
max_score = 0
best_n = 0
for n in range(1,100):
    rfc_scr = 0.
    rfc = RandomForestClassifier(n_estimators=n)
    for train, test in KFold(len(train_data), n_folds=10, shuffle=True):
        rfc.fit(train_data[predictors].T[train].T, train_data["Class"].T[train].T)
        rfc_scr += rfc.score(train_data[predictors].T[test].T, train_data["Class"].T[test].T)/10
    if rfc_scr > max_score:
        max_score = rfc_scr
        best_n = n
print(best_n, max_score)

rfc = RandomForestClassifier(n_estimators=best_n)
print('Creating submission...')
create_submission(rfc, train_data, test_data, to_check, predictors, "predictions.csv")
print('Submitted.')