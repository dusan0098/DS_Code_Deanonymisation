import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification



def random_forest_classifier(X, y, n_estimators = 300):
    train_accs = []
    valid_accs = []
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2) #n_splits=3
    
    X.fillna(0, inplace=True)
    
    for index, (train_index, valid_index) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        model = RandomForestClassifier(n_estimators=300, random_state=0)  #n_estimators=100

        model.fit(X_train, y_train)

        y_pred = model.predict(X_train).squeeze()
        train_acc = np.average(y_train == y_pred)

        y_pred = model.predict(X_valid).squeeze()
        valid_acc = np.average(y_valid == y_pred)

        print(f'Validation #{index + 1}')
        print(f'Train accuracy: {train_acc:.2f}')
        print(f'Valid accuracy: {valid_acc:.2f}\n')
        
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        
    return train_accs, valid_accs


def catboost_classifier(X, y, rsm=0.01, iterations=500):
    train_accs = []
    valid_accs = []
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    for index, (train_index, valid_index) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        model = CatBoostClassifier(
            iterations=iterations, 
            learning_rate=0.2,
            rsm=rsm,
            depth=3,
            bootstrap_type='Bernoulli',
            subsample=0.7,
            loss_function='MultiClass'
        )

        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), plot=False, verbose=False)

        y_pred = model.predict(X_train).squeeze()
        train_acc = np.average(y_train == y_pred)

        y_pred = model.predict(X_valid).squeeze()
        valid_acc = np.average(y_valid == y_pred)

        print(f'Validation #{index + 1}')
        print(f'Train accuracy: {train_acc:.2f}')
        print(f'Valid accuracy: {valid_acc:.2f}\n')
        
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        
    return train_accs, valid_accs