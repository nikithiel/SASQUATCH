# User Manual of SASQUATCH
***S***ensitivity ***A***nalysi***s*** and Uncertainty ***QUA***n***T***ification in ***C***ardiac ***H***emodynamics

*A framework for sensitivity analysis and uncertainty quantification in cardiac hemodynamics*

**Date:** June 2024\
**Mail:** thiel@ame.rwth-aachen.de or neidlin@ame.rwth-aachen.de
## A) General Idea
This project consists of three major parts:
1. Data Analysis
2. Surrogate Model Comparison
3. Sensitivity Analysis and Uncertainty Quantification
### Data Analysis 
Performing **Data Analysis** you get insights in the data you're using. That can help to find outliers and to check whether the program understands the data. You get different plots:
- Boxplot of distribution of data in columns
- Correlation Matrix of data
### Surrogate Model Comparison 
Surrogate Model Comparison gives insights on how well different Surrogate Models perform with predicting the data. To verify their behavior you get the following plots:
- mean of the timings (training and testing) over a certain number of folds
- $R^2$-score, Mean Absolut Error and Root Mean Squared Error for each model and output parameter (colors)
- Actual vs Predicted values for each output parameter (subfigure) and model (color)
All models are being saved using pickle. Use pkl.load() and model.predict() to use the models in another coding project.
### Sensitivity Analysis 
The Sensitivity Analysis provides more insights in th dependencies between input and output parameter. For the chosen models you get a heat-map which shows how sensitive a certain output parameter depends on the inputs:
- heatmap with sensitivities
Additionally it is possible to perform the sensitivity analsys with a changing bounds range. With that one can perform **uncertainty quantification**. It can be run with *(uq)*.
### Project Specific *(ps)*
It is also possible to add a project specific program.

## B) Using for Your Own Project
### General usage
1. Provide your data
2. Set preferences in *config.txt*
3. Run program and choose *da*, *sc*, *sa*, *uq*, or *ps*
4. don't worry be happy ;)
### Adding project specific features
**Adding surrogate models:**
- go to `models.py` and add a new model class according to the `NIPCE` class
- you need a `init()`, `fit()`, `predict()`, `get_params()` and `set_params()` function
- add the new model in the `creatingModels()`

**Adding preprocessing function:**
- define in `preprocessing.py` a new function `project_specific_preprocessing()`
- in `initialization.py` you can add it in `read_data()`
- you can do it like `mv_uq_procect_preprocessing()`

**Adding hyperparameters:**
- define your parameter in `config.txt`
- add the parameter in `main.py` like the others in the section `Initialize Hyperparameter` with: `your_param = parameter['your_param']`
- now you can use `your_param` in the main class

## C) Settings and Preferences
There are a couple of preferences you can set. You write the name of the variable and its value(s) separated with spaces. The order is arbitrary. It ignores text after a `#`.

Here is a short description:
### Data
| Name | Example input | Note |
| ---- | ---- | ---- |
| run_type | 'su' | use any of da, sc, sa, uq, or ps |
| data_path | `data_df.scv` *or* `../03_Results` | .xlsx, .csv, and ansys .out files |
| input_parameter | `y z alpha` | use column names of .csv/.xlsx |
| input_units | `mm mm °` | units of input parameter |
| input_parameter_label | `$y_d$ $z_d$ $\alpha$ $R_L$` | specify if you want labels that differ from input parameter names in .csv |
| output_parameter | `energy-loss wss` | use column names of .csv/.xlsx |
| output_units | `Pa m^3 Pa` | units of output parameter |
| output_parameter_label | `Eloss WSS` | specify if you want labels that differ from output parameter names in .csv |
| output_name | example | define the name of the output folder |
| is_transient | `True` | whether data is transient or not. Reduced data saved in `test_after_prep.csv` |
| lower_bound | `720` | lower bound of time steps to keep |
| upper_bound | `1200` | upper bound of time steps to keep |
| normalize | `True` | normalizing data |
| scaler | `'none'` *or* `'minmax'` *or* `'standard'` | scale data |
| save_data | `True` | save the data in .csv file |
| get_mean_of_ each_input_pair | `True` | mean over e.g. timesteps in data set. Averaged data saved in `reduced_data.csv` |

### Models
| Name | Example input | Explanation |
| ---- | ---- | ---- |
| models | `Svr-Rbf` | Support Vector Regression - Radial Basis Function |
| models | `Svr-Linear` | Support Vector Regression - Linear Basis Function |
| models | `Svr-poly` | Support Vector Regression - Polynomial Basis Function |
| models | `Svr-Sigmoid` | Support Vector Regression - Sigmoid Basis Function |
| models | `RF` | Random Forrest |
| models | `KNN` | K Nearest Neighbors |
| models | `LR` | Linear Regression |
| models | `Bayesian-Ridge` | Bayesian-Ridge |
| models | `NIPCE` | Non intrusive polynomial chaos expansion |
| models | `GP` | Gaussian Process |
| models | `DecisionTree` | Decision Tree |
| NIPCE_order | `1 2 3 4` | Specify one or multiple orders for NIPCE model |

### Training and Testing
| Name | Example input | Explanation |
| ---- | ---- | ---- |
| n_splits | `10` | number of splits for k-cross fold validation |
| shuffle | `True` | for random order of datapoints |
| random_state | `42` | for split desicion |

### Plotting
| Name | Example input | Note |
| ---- | ---- | ---- |
| plot_data | `True` | pair-plot (scatter) of data frame (currently not used)|
| is_plotting_... | `True` | specify if plotting should be utilized or not |

### Sensitivity Analysis
| Name | Example input | Explanation |
| ---- | ---- | ---- |
| sa_models | `NIPCE GP` | defines with which model(s) you want to perform the sa |
| sa_sobol_indice | `ST` *or* `S1` | Total order or first order sa |
| sa_17_segment_model | `NIPCE` | Defiens the model for segment plot |
| sa_sample_size | `512` | sample size for SA |
| sa_output_parameter | `WSS Eloss ...` | defines the output parameter for SA calculation |
| output_parameter_sa_plot | `WSS Eloss ...`| defines output parameters for plotting in GSA |
| output_units_sa_plot | `Pa m^3 Pa` | units of output parameter for plotting in GSA |
| output_parameter_sa_plot_label | `WSS Eloss ...` | specify if you want labels that differ from output parameter names in .csv |
### Project specific
Here you can add your project specific settings. In case of the *Mitral Valve Uncertainty Quantification* they're the following

| Name | Example input | Explanation |
| ---- | ---- | ---- |
| sa_17_segment_model | `Lin-Reg` | which model is used for the 17-segments plot  |
## D) Requirements
In this project you need to install the following libraries:\
**Python:** Version 3.9.13

| Library | Version |
| ---- | ---- |
| numpy | 1.25.2 |
| pandas | 2.0.0 |
| matplotlib.pyplot | 3.8.2 |
| seaborn | 0.13.0 |
| scipy | 1.12.0 |
| scikit-learn | 1.3.2 |
| chaospy | 4.3.13 |
| times |  |
| SALib |  |
| statsmodels |  |

## E) Application
This tool was used in the following publication:

*Quantifying the Impact of Mitral Valve Anatomy on Clinical Markers Using Surrogate Models and Sensitivity Analysis* \
https://engrxiv.org/preprint/view/3785

The input/output pairs used for training the surrogate models were create using Ansys Fluent CFD simulations. More details on using this automated CFD model and the corresponding setup files can be found here:

https://doi.org/10.5281/zenodo.12519189

https://www.youtube.com/watch?v=gO0ZYzpblLA
