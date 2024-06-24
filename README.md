# User Manual of SASQUATCH
***S***ensitivity ***A***nalysi***s*** and Uncertainty ***QUA***n***T***ification in ***C***ardiac ***H***emodynamics
*A framework for sensitivity analysis and uncertainty quantification in cardiac hemodynamics*

**Date:** February 2024\
**Mail:** thiel@ame.rwth-aachen.de or neidlin@ame.rwth-aachen.de\
## A) General Idea
As the name of the program already indicates, this project consists of three major parts:
1. ***D***ata ***A***nalysis
2. ***SU***rrogate Model Comparison
3. ***S***ensitivity ***A***nalysis and uncertainty quantification
### Data Analysis *(da)*
Performing **Data Analysis** you get insights in the data you're using. That can help to find outliers and to check whether the program understands the data. You get different plots:
- Boxplot of distribution of data in columns
- Correlation Matrix of data
### Surrogate Model Comparison *(sc)*
Surrogate Model Comparison gives insights on how well different Surrogate Models perform with predicting the data. To verify their behavior you get the following plots:
- mean of the timings (training and testing) over a certain number of folds
- $R^2$-score, Mean Absolut Error and Root Mean Squared Error for each model and outputparameter (colors)
- Actual vs Predicted values for each output Parameter (subfigure) and Model (color)
### Sensitivity Analysis *(sa)*
The Sensitivity Analysis provides more insights in th dependencies between input and output parameter. For the chosen models you get a heat-map which shows how sensitive a certain output parameter depends on the inputs:
- heat-map with sensitivities
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
| data_path | `data_df.scv` *or* `../03_Results` | .xlsx, .csv, and ansys .out files |
| input_parameter | `y z alpha` | use column names of .csv/.xlsx |
| output_parameter | `energy-loss wss` | use column names of .csv/.xlsx |
| output_parameter _sa_plot | `Energy_loss WSS` | names for the |
| normalize | `True` | normalizing data |
| scaler | `'none'` *or* `'minmax'` *or* `'standard'` | scale data |
| save_data | `True` | save the data in .csv file |
| get_mean_of_ each_input_pair | `True` | mean over e.g. timesteps in data set |

*To define the outputfolder go to `plotting.py` ll. 17 and specify the `save_fig_path` variable.*

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
| nipce_order | `2` | Order of NIPCE Model |

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
| is_plotting_... | `True` | currently not used |

### Sensitivity Analysis
| Name | Example input | Explanation |
| ---- | ---- | ---- |
| sa_models | `NIPCE GP` | defines with which model(s) you want to perform the sa |
| sa_sobol_indice | `ST` *or* `S1` | Total order or first order sa |
| sa_17_segment_model | `NIPCE` | Defiens the model for segment plot |
| sa_sample_size | `512` | sample size for SA |
| sa_output_parameter | `WSS Eloss ...` | defines the output parameter for SA calculation |
| output_parameter_sa_plot | `WSS Eloss ...`| defines outputs for plotting |
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
| time |  |
| warnings |  |