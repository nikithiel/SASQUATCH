from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.tree import DecisionTreeRegressor
import chaospy as cp
import pandas as pd
import numpy as np
import os

def creatingModels(model_names, bounds, parameter):
    """Creates models.

    Args:
        - model_names: list -> list of model names
        - bounds: dict -> dictionary with input-parameter bounds
    Returns:
        - models: dict ->  dictionary containing the models
        - final_model_names: list -> list of strings with model names for plots
    """
    # Dictionary with models:
    models = {}
    implemented_models = ['Svr-Rbf', 'Svr-Linear', 'Svr-Poly', 'Svr-Sigmoid', 'RF', 'KNN', 'LR', 'Bayesian-Ridge', 'NIPCE','GP', 'DecisionTree']
    
    final_model_names = []
    if 'Svr-Rbf' in model_names:
        models['SVR-rbf'] = SVR(kernel='rbf', C = 150, gamma=0.1,epsilon=0.1,tol=1e-5)  # Radial Basis Function (RBF) kernel
        final_model_names.append('SVR-rbf')
    if 'Svr-Linear' in model_names:
        models['SVR-linear'] = SVR(kernel='linear')  # linear kernel
        final_model_names.append('SVR-linear')
    if 'Svr-Poly' in model_names:
        models['SVR-poly'] = SVR(kernel='poly')  # poly kernel
        final_model_names.append('SVR-poly')
    if 'Svr-Sigmoid' in model_names:
        models['SVR-sigmoid'] = SVR(kernel='sigmoid')  # sigmoid kernel
        final_model_names.append('SVR-sigmoid')
    if 'RF' in model_names:
        models['RF'] = RandomForestRegressor(n_estimators=10)  # Random Forest Regressor
        final_model_names.append('RF')
    if 'KNN' in model_names:
        models['KNN'] = KNeighborsRegressor()  # K-Nearest Neighbors Regressor
        final_model_names.append('KNN')
    if 'LR' in model_names:
        models['LR'] = LinearRegression()  # Linear Regression
        final_model_names.append('LR')
    if 'Bayesian-Ridge' in model_names:
        models['Bayesian Ridge'] = BayesianRidge(compute_score=True, n_iter=2) # Bayesian Ridge
        final_model_names.append('Bayesian Ridge')
    if 'NIPCE' in model_names:
        distributions = [cp.Uniform(variable['min'], variable['max']) for variable in bounds.values()]

        if 'NIPCE_order' in parameter:

            orders = parameter['NIPCE_order']

            if isinstance(parameter['NIPCE_order'], int):
                orders = [orders]

            for order in orders:
                nipce_name = 'NIPCE ' + str(order)
                models[nipce_name] = NIPCE(order=order, distributions=distributions)
                final_model_names.append(nipce_name)
        else:
            models['NIPCE_order_3'] = NIPCE(order=3, distributions=distributions)
            final_model_names.append('NIPCE 3')
            print("! Warning: NIPCE order not specified, using default order 3 !")
    if 'GP' in model_names:
        length_scale = [(bound['max'] - bound['min']) for bound in bounds.values()]
        # for finding a good parameter nu
        find_good_nu = False
        find_good_alpha = False
        if find_good_nu:
            for nu in np.arange(0.1, 1.6, 0.2):
                nu = np.around(nu, decimals=1)
                gp_name = str(nu)
                kernel = Matern(length_scale=length_scale, nu=nu)
                models[gp_name] = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
                final_model_names.append(gp_name)
        elif find_good_alpha:
            for alpha in np.arange(0.0, 1, 0.2):
                alpha = np.around(alpha, decimals=1)
                gp_name = str(alpha)
                kernel = Matern(length_scale=length_scale, nu=0.5)
                models[gp_name] = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=alpha)
                final_model_names.append(gp_name)
        else:
            kernel = Matern(length_scale=length_scale, nu=0.5)
            models['GP'] = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=0)
            final_model_names.append('GP')
    if 'DecisionTree' in model_names:
        models['Decition Tree'] = DecisionTreeRegressor()
        final_model_names.append('Decition Tree')

    if isinstance(model_names, str):
        model_names = [model_names]
        
    for model in model_names:
        if model not in implemented_models:
            print("! Warning: Ignoring Unknown Model: ", model," !")
    return models, final_model_names

# NIPCE Class defined here because it's not a sklearn model
class NIPCE:
    """Class for Non Intrusive Polynomial Chaos Expansion model."""

    def __init__(self, order=3, distributions=None):
        """Initialization of model.
        
        Args:
            - order: int -> order of NIPCE model
            - distribution: distribution -> distribution for expansion
        """
        self.order = order
        self.distributions = distributions
        self.expansion = cp.generate_expansion(self.order, cp.J(*self.distributions))
        self.model = None

    def fit(self, X, y):
        """Fit the NIPCE model using chaospy.

        Args:
            - X: dataFrame or numpy_array -> input data
            - y: dataFrame or numpy_array -> ouput data
        Returns:
            self.model: NIPCE -> trained model
        """
        if isinstance(X,  pd.DataFrame): X = X.values
        self.model = cp.fit_regression(self.expansion, X.T, y)

        return self.model

    def predict(self, X_test):
        """Test the trained NIPCE model.

        Args:
            X_test: dataFrame or numpy_array -> input data
        Returns:
            prediciton: numpy_array -> prediction of test data
        """
        if isinstance(X_test, pd.DataFrame): X = X_test.values
        else: X = X_test
        prediction = self.model(*(X.T))
        return prediction.T
    
    def get_params(self, deep=True):
        """Return parameters as a dictionary.

        Returns:
            dictionary -> contains order and distributions
        """ 
        return {'order': self.order, 'distributions': self.distributions}

    def set_params(self, **params):
        """Sets Parameters.

        Args:
            params -> parameters
        """
        # Set parameters from the input dictionary
        self.order = params.get('order', self.order)
        self.bounds = params.get('distributions', self.distributions)
        return self