import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error

def kFold_Evaluation(X, y, n_splits, shuffle, random_state, models):
    """Performs kFold cross validation.
    
    Args:
        - X: dataFrame -> contains input data
        - y: dataFrame -> contains ouput data
        - n_splits: int -> number of folds
        - shuffle: bool -> whether shuffeling the data for splitting
        - random_state: int -> which random state for splitting
        - models: dict -> contains surrogate models
    Returns:
        dataFrame -> contains results of R2Score, Timings, MAE, and RMSE
    """
    # creating k-Fold
    if random_state=='rand': random_state = np.random.randint(1000)
    k_fold = KFold(n_splits=n_splits, shuffle=shuffle, random_state = random_state)
    fold_results_list = []
    
    for model_name, model in models.items():
        for fold, (train_index, test_index) in enumerate (k_fold.split(X), 1):
            # Training and Predicting
            start_time = time.time()
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            fitted_model = MultiOutputRegressor(model).fit(X_train,y_train)
            y_pred_array = fitted_model.predict(X_test)
            y_pred = pd.DataFrame(y_pred_array, columns=y.columns)
            end_time = time.time()
            
            # calculating r2 score and errors
            metrics = {}
            for column in y.columns:
                r2 = r2_score(y_test[column].values, y_pred[column].values)
                mae = mean_absolute_error(y_test[column].values, y_pred[column].values)
                rmse = np.sqrt(mean_squared_error(y_test[column].values, y_pred[column].values))
                mape = mean_absolute_percentage_error(y_test[column].values, y_pred[column].values)
                maxe = max_error(y_test[column].values, y_pred[column].values)
                metrics[f'r2_score_{column}'] = r2
                metrics[f'MAE_{column}'] = mae
                metrics[f'RMSE_{column}'] = rmse
                metrics[f'MAPE_{column}'] = mape
                metrics[f'MAXE_{column}'] = maxe

            # saving results in the dataframe
            fold_results = {
                'Fold': fold,
                'Model': model_name,
                'Timing': end_time - start_time,
                **metrics
            }
            fold_results_list.append(fold_results)
        print("    Training of ",model_name,": Done")
            
    return pd.DataFrame(fold_results_list)