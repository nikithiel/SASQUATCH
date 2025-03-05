import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error
import pickle as pkl
import os

def kFold_Evaluation(X, y, n_splits, shuffle, random_state, models, folder):
    """Performs kFold cross validation.
    
    Args:
        - X: dataFrame -> contains input data
        - y: dataFrame -> contains ouput data
        - n_splits: int -> number of folds
        - shuffle: bool -> whether shuffeling the data for splitting
        - random_state: int -> which random state for splitting
        - models: dict -> contains surrogate models
        - folder: str -> folder to save results
    Returns:
        dataFrame -> contains results of R2Score, Timings, MAE, and RMSE
    """

    # Creating k-Fold
    if random_state=='rand': random_state = np.random.randint(1000)
    k_fold = KFold(n_splits=n_splits, shuffle=shuffle, random_state = random_state)
    fold_results_list = []
    
    for model_name, model in models.items():
        for fold, (train_index, test_index) in enumerate (k_fold.split(X), 1):
            # Training and Predicting
            start_time = time.time()
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            fitted_model = MultiOutputRegressor(model).fit(X_train, y_train)

            y_pred_array = fitted_model.predict(X_test)
            y_pred = pd.DataFrame(y_pred_array, columns=y.columns)
            end_time = time.time()
            
            # Calculating r2 score and errors
            metrics = {}
            for column in y.columns:
                r2 = r2_score(y_test[column].values, y_pred[column].values)
                mae = mean_absolute_error(y_test[column].values, y_pred[column].values)
                rmse = np.sqrt(mean_squared_error(y_test[column].values, y_pred[column].values))
                mape = mean_absolute_percentage_error(y_test[column].values, y_pred[column].values, multioutput='uniform_average')
                maxe = max_error(y_test[column].values, y_pred[column].values)
                metrics[f'r2_score_{column}'] = r2
                metrics[f'MAE_{column}'] = mae
                metrics[f'RMSE_{column}'] = rmse
                metrics[f'MAPE_{column}'] = mape
                metrics[f'MAXE_{column}'] = maxe

            # Saving results in the dataframe
            fold_results = {
                'Fold': fold,
                'Model': model_name,
                'Timing': end_time - start_time,
                **metrics
            }
            fold_results_list.append(fold_results)
            
    return pd.DataFrame(fold_results_list)

def train_and_save_models(X, y, models, folder):
    """
    Train models on the full dataset and save them to disk.

    Args:
        - X: dataFrame -> contains input data
        - y: dataFrame -> contains ouput data
        - models: dict -> contains surrogate models
        - folder: str -> folder to save results
    """

    # Ensure the output folder exists
    os.makedirs(folder, exist_ok=True)

    # Iterate over each model
    for model_name, model in models.items():
        # Train the final model on the full dataset
        fitted_model = MultiOutputRegressor(model).fit(X, y)
        model_file = model_name + '.pkl'
        model_path = os.path.join(folder, model_file)

        with open(model_path, 'wb') as f:
                pkl.dump(fitted_model, f)

        if model_name == 'NIPCE':

            polynomFile = os.path.join(folder, 'NIPCE_order_' + str(fitted_model.estimator.order) + '.txt')

            for i, estimator in enumerate(fitted_model.estimators_):

                with open(polynomFile, 'a') as file:
                    
                    file.write(str(estimator.model.round(12)) + '\n\n')

        print("    Training of ", model_name, ": Done")
        print(f"    {model_name} saved to {model_path}")

        ### TESTING ONLY ###
        if model_name == 'GP':
            
            csv = os.path.join('input_data','Data_example.csv')
            df = pd.read_csv(csv)

            model_NIPCE_2 = pkl.load(open(os.path.join(folder, 'NIPCE_order_2.pkl'), 'rb'))
            model_NIPCE_4 = pkl.load(open(os.path.join(folder, 'NIPCE_order_4.pkl'), 'rb'))
            model_LR = pkl.load(open(os.path.join(folder, 'LR.pkl'), 'rb'))
            model_GP = pkl.load(open(os.path.join(folder, 'GP.pkl'), 'rb'))

            sample = df.sample()
            print(sample)
            
            sample_input = sample.iloc[:, 0:4].copy()
            res_NIPCE_2 = model_NIPCE_2.predict(sample_input)
            res_NIPCE_4 = model_NIPCE_4.predict(sample_input)
            res_LR = model_LR.predict(sample_input)
            res_GP = model_GP.predict(sample_input)
            print('NIPCE 2:', res_NIPCE_2)
            print('NIPCE 4:', res_NIPCE_4)
            print('RL:', res_LR)
            print('GP:', res_GP)
            print('Testing done!')