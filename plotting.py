import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import seaborn as sns
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import ScalarFormatter
import os
from sklearn.multioutput import MultiOutputRegressor
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import ticker
from scipy.stats import gaussian_kde
from statsmodels.multivariate.manova import MANOVA

thesis = False

plt_style = { # Use to change the formatting and text of the plots
    'mathtext.fontset' : 'stix', 
    'font.family' : 'STIXGeneral', 
    'font.size' : 10
    }
plt.rcParams.update(**plt_style)
# Read specifically the types to be plotted from the input file
config_file = 'config.txt'
with open(config_file,'r') as file:
    for line in file:
        # Read plot type
        if line:
            splitted_line = line.split()
        if splitted_line == []:
            continue
        
        if splitted_line[0] == 'plot_type':
            #splitted_line = line.split()
            values_end = next((i for i, x in enumerate(splitted_line) if x.startswith('#')), None)
            if len(splitted_line) > 1:
                values = splitted_line[1:values_end]

                # Store the values as a list
                types = []
                for value in values:
                    types.append(value)

        # Read output parameters
        if splitted_line[0] == 'output_parameter':
            #splitted_line = line.split()
            values_end = next((i for i, x in enumerate(splitted_line) if x.startswith('#')), None)
            if len(splitted_line) > 1:
                values = splitted_line[1:values_end]

                # Store the values as a list
                output_parameter_list = []
                for value in values:
                    output_parameter_list.append(value)
                    
        # Read output units
        if splitted_line[0] == 'output_units':
            #splitted_line = line.split()
            values_end = next((i for i, x in enumerate(splitted_line) if x.startswith('#')), None)
            if len(splitted_line) > 1:
                values = splitted_line[1:values_end]

                # Store the values as a list
                output_unit_list = []
                for value in values:
                    output_unit_list.append(value)

        # Read output labels
        if splitted_line[0] == 'output_parameter_label':
            #splitted_line = line.split()
            values_end = next((i for i, x in enumerate(splitted_line) if x.startswith('#')), None)
            if len(splitted_line) > 1:
                values = splitted_line[1:values_end]

                # Store the values as a list
                output_label_list = []
                for value in values:
                    output_label_list.append(value)
        
        # Read input parameters
        if splitted_line[0] == 'input_parameter':
            #splitted_line = line.split()
            values_end = next((i for i, x in enumerate(splitted_line) if x.startswith('#')), None)
            if len(splitted_line) > 1:
                values = splitted_line[1:values_end]

                # Store the values as a list
                input_parameter_list = []
                for value in values:
                    input_parameter_list.append(value)
        # Read input units
        if splitted_line[0] == 'input_units':
            #splitted_line = line.split()
            values_end = next((i for i, x in enumerate(splitted_line) if x.startswith('#')), None)
            if len(splitted_line) > 1:
                values = splitted_line[1:values_end]

                # Store the values as a list
                input_unit_list = []
                for value in values:
                    input_unit_list.append(value)

        # Read input labels
        if splitted_line[0] == 'input_parameter_label':
            #splitted_line = line.split()
            values_end = next((i for i, x in enumerate(splitted_line) if x.startswith('#')), None)
            if len(splitted_line) > 1:
                values = splitted_line[1:values_end]

                # Store the values as a list
                input_label_list = []
                for value in values:
                    input_label_list.append(value)

input_dict = dict(zip(input_parameter_list, input_label_list))
output_dict = dict(zip(output_parameter_list, output_label_list))
class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self, vmin=None, vmax=None):

        self.format = "%1.1f"  # Force one decimal place in the tick labels.

# The mixed function for every runtype
def plot_data_analysis(X_df, y_df, output_path, **kwargs):
    plot_correlation(X_df, y_df, output_path)
    plot_feature_distribution(y_df, output_path, dict(zip(kwargs['output_parameter'],kwargs['output_units'])), num_bins = 20, title="Distribution of Output Parameter")
    actual_scatter_plot(X_df, y_df, output_path)

    # Optional plots
    plot_boxplots(X_df, y_df, output_path, title="Boxplots of Ouput Parameter") if input(" Would you like to plot the boxplot as well?: (y/n)") == 'y' else None
    plot_data(X_df, y_df, output_path) if input(" Would you like to plot the detailed scatterplot as well?: (y/n)") == 'y' else None

def plot_surrogate_model_comparison(smc_results, X_df, y_df, output_path, models, kwargs):
    plot_smc_r2score_and_errors(smc_results, kwargs['output_parameter'], output_path, metrics=kwargs['metrics'], average_output=True, is_title=False, title="Model Errors and Scores avrg r2 mape")
    plot_smc_r2score_and_errors(smc_results, kwargs['output_parameter'], output_path, metrics=kwargs['metrics'], average_output=False, is_title=False, title="Model Errors and Scores", rmse_log_scale=True) if input(" Would you like to plot detailed r2 score?: (y/n) ") == 'y' else None
    surrogate_model_predicted_vs_actual(models, X_df, y_df, output_path, kwargs['output_parameter'], dict(zip(kwargs['output_parameter'], kwargs['output_units']))) if input(" Would you like to plot the model comparison as a scatter plot?: (y/n) ") == 'y' else None
    plot_smc_timings(smc_results, output_path, is_title=False) if input(" Would you like to plot the surrogate model comparison timing plot?: (y/n) ") == 'y' else None

def plot_sensitivity_analysis(X_dict, y_dict, output_path, sa_results, model_names, input_parameter_list, parameter):
    plot_densities(X_dict, y_dict, output_path)
    plot_sa_results_heatmap(sa_results, model_names, input_parameter_list, parameter['output_parameter_sa_plot'], output_path, parameter['sa_sobol_indice'])
    plot_sa_results_17_segments(sa_results, input_parameter_list, output_path, parameter['sa_17_segment_model'], parameter['sa_sobol_indice']) if input (" Plot the 17 segment model as well? (y/n)") == 'y' else None

# Individual Plotting functions

""" ----- For Data Analysis ----- """
def plot_correlation(X_df, y_df, output_path, is_title=True, title="Correlation Matrix"):
    """Plots and saves the correlation matrix of a dataframe.
    Args:
        - X_df: dataFrame -> contains the data for inputs
        - y_df: dataFrame -> contains the data for outputs
        - output_path: str -> path to save the plot
        - title: st -> title of plot and name of saved figure
    """
    # Compute the correlation matrix
    df_combined = pd.concat([X_df, y_df], axis=1)
    correlation_matrix = df_combined.corr()

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10.,10.))
    sns.heatmap(correlation_matrix, cmap='viridis', fmt=".2f", xticklabels = input_label_list + output_label_list, yticklabels = input_label_list + output_label_list)
    plt.title(title if is_title else None)
    plt.xticks(rotation=45)
    save_plot(plt, output_path + title)

def plot_feature_distribution(df, output_path, output_units=None, num_bins=10, is_title=True, title="Plot of Feature Distributions", num_subplots_in_row=3, figure_size='large'):
    """Plots and saves the distribution of each feature of the dataFrame.
    Args:
        - df: dataFrame -> contains data
        - output_path: str -> path to save the plot
        - output_units: dict -> dictionary of units for each output parameter
        - num_bins: int -> number of bins
        - title: str -> title of plot / name of saved figure
    """

    if figure_size == 'large':
        fig_size=(6.5,9)
    elif figure_size == 'small':
        fig_size=(6.5,4)

    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    num_plots = len(num_columns)
    num_cols = (num_plots - 1) // num_subplots_in_row + 1  # Calculate the number of rows needed
    num_rows = min(num_plots, num_subplots_in_row)  # Limit the number of columns to 4

    fig, axes = plt.subplots(num_cols, num_rows, figsize=fig_size)
    axes = axes.flatten()

    k = 0
    for i, col in enumerate(num_columns):
        axes[i].hist(df[col], bins=num_bins)

        if output_unit_list is not None:
            axes[i].set_xlabel(output_unit_list[k], labelpad=12)
            axes[i].set_title(output_label_list[k], pad=12)
            k += 1
        else:
            axes[i].set_xlabel(col)
        if i % num_rows == 0:
            axes[i].set_ylabel('Frequency')

    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title if is_title else None)
    plt.tight_layout()
    save_plot(plt, output_path + title)

def actual_scatter_plot(X_df, y_df, output_path, is_title=True, title="Pairplot of Data reduced"):
    """Plot a reduced pairplot of the data

    Args:
        X_df (DataFrame): Parameter input data.
        y_df (DataFrame): Parameter output data.
        output_path (str): Path where the file will be saved.
        is_title (bool, optional): True to plot title. Defaults to True.
        title (str, optional): Title name. Defaults to "Pairplot of Data reduced".
    """
    plot_size = max(X_df.shape[1], y_df.shape[1])
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'ST'
    plt.ticklabel_format(axis='y', style='sci', scilimits= (0,0))
    fig, axes = plt.subplots(y_df.shape[1], X_df.shape[1], figsize=(50.,50.))
    i = 0
    j = 0
    k = 0
    k2 = 0
    combined_labels = input_label_list + output_label_list
    for names, values in y_df.items():
        for names2, values2 in X_df.items():
            if j == 0:
                axes[i][j].set_ylabel(combined_labels[k], fontsize=30)
                k += 1
            if i == y_df.shape[1] - 1:
                axes[i][j].set_xlabel(input_label_list[k2], fontsize=30)
                k2 += 1
            axes[i][j].scatter(X_df[names2].values, y_df[names].values, marker='.')
            axes[i][j].ticklabel_format(axis='y', style='sci', scilimits= (0,0))
            j += 1
        i += 1
        j = 0

    fig.tight_layout()
    save_plot(plt=plt, file_path=output_path + title, dpi=600)
    
def plot_boxplots(X_df, y_df, output_path, is_title=True, title="Boxplots of Dataframe"):
    """Plots and saves the boxplot of a dataframe.
    Args:
        - X_df: DataFrame -> contains the data for inputs
        - y_df: DataFrame -> contains the data for boxplots
        - output_path: str -> path to save the plot
        - title: str -> title of plot and name of saved figure
    """
    num_columns = int(math.sqrt(y_df.shape[1])) + 1
    #plt.ticklabel_format(axis='y', style='sci', scilimits= (0,0))
    fig, axes = plt.subplots(nrows=num_columns, ncols=num_columns, figsize=(10,10))

    i = 0
    j = 0
    k = 0 # Counter for labels
    for _, column in enumerate(y_df.columns):
        ax = axes[i][j] if num_columns > 1 else axes
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2e'))
        ax.set_title(output_label_list[k])
        ax.ticklabel_format(axis='y', style='sci', scilimits= (0,0))
        ax.boxplot(y_df[column])
        ax.set_ylabel('Values in '+ output_unit_list[k])
        ax.set_xticklabels([], rotation=0)
        if j < num_columns - 1:
            j += 1
        elif j == num_columns - 1:
            i += 1
            j = 0
        k += 1
    
    # Delete all remaining empty subplots
    if j != num_columns - 1: 
        while i <= num_columns - 1:
            while j <= num_columns - 1:
                axes[i][j].set_axis_off()
                j += 1 
            i += 1
            j = 0

    plt.suptitle(title if is_title else None)
    plt.tight_layout()
    save_plot(plt, output_path + title)

def plot_data(X_df, y_df, output_path, is_title=True, title="Pairplot of Data"):
    """Plots and saves the pairplot of data.
    Args:
        - X_df: dataFrame -> contains input data
        - y_df: dataFrame -> contains output data
        - output_path: str -> path to save the plot
        - title: str -> title of plot / name of saved figure
    """
    # Create a DataFrame combining selected features and outputs
    pairplot = sns.pairplot( pd.concat([X_df, y_df], axis=1))
    pairplot.x_vars = input_label_list + output_label_list
    pairplot.y_vars = input_label_list + output_label_list
    pairplot._add_axis_labels()
    for ax in pairplot.axes.flatten():
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    save_plot(plt, output_path + title, dpi=600)

""" ----- For Surrogate Model Comparison ----- """ 
def surrogate_model_predicted_vs_actual(models, X_df, y_df, output_path, output_parameter, output_units, is_title=True, title="Surrogate Model Comparison"):
    """Plots and saves the model comparison (actual vs predicted).
    Args:
        models (dict): Contains the models.
        X_df (DataFrame): Input data.
        y_df (DataFrame): Output data.
        output_path (str): Path to save the plot.
        output_parameter: List (or similar) containing output column names.
        output_units (dict): Dictionary containing units for each output parameter.
        is_title (bool): Whether to show a title on each subplot.
        title (str): Overall title of the plot / name of saved figure.
    """

    # Determine number of outputs
    output_columns = y_df[output_parameter].columns
    n = len(output_columns)

    if n == 1:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        axs = [ax]
        nrows = 1
    else:
        nrows = math.ceil(n / 2)
        fig, axs_array = plt.subplots(nrows, 2, figsize=(6.5, 5 * nrows / 2))
        axs = list(axs_array.flatten())

    # Plot each output
    for i, output_name in enumerate(output_columns):
        for model_name, model in models.items():
            fitted_model = MultiOutputRegressor(model).fit(X_df, y_df)
            y_pred_array = fitted_model.predict(X_df)
            y_pred = pd.DataFrame(y_pred_array, columns=y_df.columns)
            axs[i].scatter(y_df.iloc[:, i], y_pred.iloc[:, i],
                           label=model_name, marker='o', s=8)
        axs[i].plot(y_df.iloc[:, i], y_df.iloc[:, i],
                    label='Actual', c='black', linewidth=1.0)

        axs[i].ticklabel_format(axis='both', style='sci', scilimits=(0,0))

        # Label axes: bottom row gets the x-label; left column gets the y-label
        row, col = divmod(i, 2)

        # For even number of outputs, only the bottom row (row == nrows - 1) gets the xlabel
        # For odd number of outputs, both the last row and the row above it (if any) get the xlabel
        if n % 2 == 0:
            if row == nrows - 1:
                axs[i].set_xlabel('Actual Values')
        else:
            if nrows > 1:
                if row >= nrows - 2:
                    axs[i].set_xlabel('Actual Values')
            else:
                axs[i].set_xlabel('Actual Values')

        # The y-label is set for the left column
        if col == 0:
            axs[i].set_ylabel('Predicted Values')

        combined_labels = input_label_list + output_label_list
        combined_units = input_unit_list + output_unit_list
        if is_title:
            axs[i].set_title(f'{combined_labels[i]} in {combined_units[i]}')

        axs[i].grid(False)

    # For odd number of outputs (n > 1), remove the extra axis
    if n % 2 == 1 and n > 1:
        extra_axis = axs.pop()
        extra_axis.remove()

    # Reserve more space on the right for the legend
    # The rect parameter (left, bottom, right, top) here reserves only 75% of the width for subplots
    plt.tight_layout(rect=[0, 0, 0.83, 1])

    # Re-center the last subplot if n is odd
    if n % 2 == 1 and n > 1:
        last_axis = axs[-1]
        pos = last_axis.get_position()
        # Use the current subplot area margins to compute the center
        subplot_left = fig.subplotpars.left
        subplot_right = fig.subplotpars.right
        center = (subplot_left + subplot_right) / 2
        new_x0 = center - pos.width / 2
        last_axis.set_position([new_x0, pos.y0, pos.width, pos.height])

    # Place the legend
    # Use the top-right axis from the first row (if available) as a reference
    if n > 1:
        top_right = axs[1]
    else:
        top_right = axs[0]
    pos_top_right = top_right.get_position()
    # Increase the horizontal offset for more legend width
    legend_x = pos_top_right.x1  
    legend_y = pos_top_right.y1 + 0.015
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(legend_x, legend_y), loc='upper left')

    save_plot(plt, output_path + title)      

def plot_smc_r2score_and_errors(df, output_parameter, output_path, output_units=None, metrics=['r2_score', 'RMSE', 'MAPE'], \
                                is_title=True, title="Surrogate Model Comparison Plots", average_output=False, rmse_log_scale=False, figure_size = (6.5,4)):
    """Plots and saves the R2 score and Errors.
    Args:
        - df: dataFrame -> contains sc result data
        - output_parameter: list -> list of strings of output_parameter
        - output_path: str -> path to save the plot
        - metrics: list -> list of metrics
        - is_title: bool -> whether title is present
        - title: str -> title of plot / name of saved figure
        - rmse_log_scale: bool: whether using a log scale for rmse
    """
    # Melt the DataFrame to have a tidy format for plotting
    output_parameters = []
    if isinstance(metrics, str):
        metrics = [metrics]

    for prefix in metrics:
        for suffix in output_parameter:
            output_parameters.append(prefix + '_' + suffix)

    df_melted = pd.melt(df, id_vars=['Fold', 'Model', 'Timing'], 
                        value_vars=output_parameters, var_name='Output Parameter', value_name='Value')

    # Calculate mean R2 scores and RMSE for each Model and Metric
    mean_scores = df_melted.groupby(['Model', 'Output Parameter'])['Value'].mean().reset_index()
    mean_scores['Value'] = mean_scores['Value'].apply(lambda x: 0 if x < 0 else x)

    # Create subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.,5.))

    # Bar plot for mean R2 scores
    for i, metric in enumerate(metrics):
        # Filter mean_scores DataFrame for the current metric
        filtered_df = mean_scores[mean_scores['Output Parameter'].str.startswith(metric)].copy()

        if metric == 'MAPE': filtered_df['Value'] = filtered_df['Value'] * 100

        # remove metric string (RMSE, MAE, r2_score) for legend
        for metric_in_metrics in metrics:
            filtered_df['Output Parameter'] = filtered_df['Output Parameter'].str.replace(metric_in_metrics + '_', '')

        average_values = filtered_df.groupby('Model')['Value'].mean().reset_index()

        # Merge the average values back to the filtered_df DataFrame and sort
        filtered_df = pd.merge(filtered_df, average_values.rename(columns={'Value': 'output_average'}), on='Model')
        filtered_df = filtered_df.sort_values(by='output_average', ascending= metric!='r2_score')
        if not average_output: filtered_df = filtered_df.sort_values(by=['output_average', 'Model', 'Value'], ascending=metric!='r2_score')

        metric_title = metric
        if metric == 'r2_score': metric_title=r'R$^2$ Score'
        if metric == 'MAPE': metric_title='MAPE in %'
        if len(metrics) == 1:
            if average_output: sns.barplot(x='Model', y='Value', color='tab:blue', data=filtered_df, ax=axes, errorbar='ci')
            else: sns.barplot(x='Model', y='Value', hue='Output Parameter', data=filtered_df, ax=axes, errorbar='ci')
            # Set title, xlabel, and ylabel for the current subplot
            axes.set_title(f'{metric_title}')
            axes.set_xlabel('')
            axes.set_ylabel('')
            axes.tick_params(axis='x', rotation=0)
            if not average_output: axes.legend_.remove()
        else:
            if average_output: sns.barplot(x='Model', y='Value', color='tab:blue', data=filtered_df, ax=axes[i], errorbar='ci')
            else: sns.barplot(x='Model', y='Value', hue='Output Parameter', data=filtered_df, ax=axes[i], errorbar='ci')
            # Set title, xlabel, and ylabel for the current subplot
            axes[i].set_title(f'{metric_title}')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            if rmse_log_scale and metric == 'RMSE': axes[i].set_yscale('log')
            axes[i].tick_params(axis='x', rotation=45)
            if not average_output: axes[i].legend_.remove()
    if not average_output:
        if len(metrics) == 1:
            handles, labels = axes.get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(1, 0.5), loc='center left')
        else:
            handles, labels = axes[0].get_legend_handles_labels()
            if thesis:
                for i, label in enumerate(labels):
                    labels[i] = output_dict[label]
            fig.legend(handles, labels, bbox_to_anchor=(1, 0.5), loc='center left', fontsize='small')
    #plt.tight_layout()
    save_plot(plt, output_path + title)

def plot_smc_timings(df, output_path, is_title=True, title="Boxplot of Timings of All Folds of Each Model"):
    """Plots and saves the training and testing timings.
    Args:
        - df: dataFrame -> contains sc result data
        - output_path: str -> path to save the plot
        - title: str -> title of plot / name of saved figure
    """
    # Plotting the bar plot
    plt.figure(figsize=get_figsize())
    grouped_data = [df[df['Model'] == model]['Timing'] for model in df['Model'].unique()]
    plt.boxplot(grouped_data, labels=df['Model'].unique())
    #sns.boxplot(x='Model', y='Timing', data=df)

    # Labeling axes and title
    #plt.xlabel('Model')
    plt.ylabel('Train + Predict Time in s')
    plt.title(title if is_title else None)

    # Show plot
    plt.tight_layout()
    save_plot(plt, output_path + title)

""" ----- For Sensitivity Analysis ----- """
def plot_densities(X_dict, Y_dict, output_path, labels=None, is_title=True, title="Densities plot"):
    """Plots and saves the distribution of each feature of the DataFrame.
    Args:
        - X_dict: dict -> Dictionary containing arrays for each model's test results
        - Y_dict: dict -> Dictionary containing labels for each model
        - output_path: str -> path to save the plot
        - is_title: bool -> Whether to display the title
        - title: str -> Title of the plot / name of the saved figure
    """
    num_models = len(Y_dict)
    num_plots = len(Y_dict[list(Y_dict.keys())[0]][0])  # Assuming all models have the same number of columns
    num_cols = min(num_plots, 3)   # Limit the number of columns to
    num_rows =  (num_plots - 1) // num_cols + 1 # Calculate the number of rows needed
    num_models
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6.5, 9))
    axes = axes.flatten()
    handles, labels = [], []
    for i in range(num_plots):
        for model, data in Y_dict.items():
            try:
                kde = gaussian_kde(data[:, i])
                x = np.linspace(data[:, i].min(), data[:, i].max(),1000)
                line, = axes[i].plot(x,kde(x), label=model)
                if i==0:
                    handles.append(line)
                    labels.append(model)
                if i%num_cols==0: axes[i].set_ylabel('Density')
            except:
                pass

        axes[i].set_xlabel(output_label_list[i] +" in "+ output_unit_list[i])

    # Hide unused subplots   
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    for handle, data in zip(handles, Y_dict.values()):
        #handle.set_color(axes[0].lines[Y_dict.keys().index(data)].get_color())
        #handle.set_color(axes[0].get_color())
        handle.set_color(handle.get_color())

    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.01), loc='upper center', ncol=len(Y_dict))    

    plt.suptitle(title if is_title else None)
    plt.tight_layout()
    save_plot(plt, output_path+title)

def plot_feature_scatterplot(df, x_cols, y_cols, output_path, fig_size=(6.5, 6.5), is_title=True, title="Scatterplot"):
    """Plots and saves scatterplot of two columns from the DataFrame.
    
    Args:
        - df: DataFrame -> contains data
        - x_cols: list -> names of the columns for x-axis
        - y_cols: list -> names of the columns for y-axis
        - output_path: str -> path to save the plot
        - title: str -> title of plot
    """

    n_plots = len(x_cols) * len(y_cols)

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 10

    fig, axes = plt.subplots(len(y_cols), len(x_cols), figsize=fig_size)

    for i, y_col in enumerate(y_cols):

        # Create mask for quantifying outliers based on 1.5 IQR
        y_data = df[y_col].dropna()

        # Get exponent for custom ScalarFormatter of the y-axis.
        if not y_data.empty:
            median_val = np.median(y_data)
            # Avoid log10(0) and compute exponent from absolute median value
            exponent = int(np.floor(np.log10(np.abs(median_val)))) if median_val != 0 else 0
        else:
            exponent = 0  # Default exponent if no data

        Q1 = y_data.quantile(0.25)
        Q3 = y_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df[y_col] < lower_bound) | (df[y_col] > upper_bound)

        for j, x_col in enumerate(x_cols):
            ax = axes[i, j] if n_plots > 1 else axes

            inliers_df = df[~outlier_mask]
            outliers_df = df[outlier_mask]

            ax.scatter(inliers_df[x_col], inliers_df[y_col],
                       s=8, label='Inliers')
            ax.scatter(outliers_df[x_col], outliers_df[y_col],
                       s=8, c='orange', label='Outliers')

            if i == len(x_cols)-1: ax.set_xlabel(x_cols[x_col]['label'] + " in " + x_cols[x_col]['unit'])
            if j == 0: ax.set_ylabel(y_cols[y_col]['label'] + " in " + y_cols[y_col]['unit'])

            if is_title:
                ax.set_title(f"{title} - {x_cols[x_col]['label']} vs {y_cols[y_col]['label']}")

            # Apply the custom ScalarFormatter to the y-axis.
            formatter = ScalarFormatterForceFormat()
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    save_plot(plt, output_path + title)

def plot_sa_results_heatmap(sa_results, model_names, input_parameter_list, output_parameter_sa_plot, output_path, sa_sobol_indice):
    """Plots and saves the sensitivity heatmap.
    Args:
        - sa_results: dict -> contains sensitivity analysis results
        - model_names: list -> list of names of models
        - input_parameter_list: list -> list of input parameter
        - output_parameter_sa_plot: list -> list of output parameters for sa plot
        - output_path: str -> path to save the plot
        - sa_sobol_indice: str -> Sobol indice for sa (ST or S1)
    """
    input_parameter_list = list(input_parameter_list)
    if thesis:
        for i, input in enumerate(input_parameter_list):
            input_parameter_list[i] = input_dict[input]
    fig, axes = plt.subplots(1, len(model_names), figsize=get_figsize())
    title = sa_sobol_indice+" Sensitivity Analysis"
    # Plot heatmap for each model
    for i, (model, model_dict) in enumerate(sa_results.items()):
        sensitivity_values = []
        output_parameter_list = []
        for (output_parameter, sensitivity_dict) in model_dict.items():
            if model in model_names and sa_sobol_indice in list(sensitivity_dict.keys()) and output_parameter in output_parameter_sa_plot:
                sensitivity_values.append(sensitivity_dict[sa_sobol_indice])
                if thesis: output_parameter_list.append(output_dict[output_parameter])
                else:  output_parameter_list.append(output_parameter)
        #print(sensitivity_values)
        if sensitivity_values:
            if len(model_names) == 1:
                sns.heatmap(np.array(sensitivity_values), cmap="viridis", ax=axes, xticklabels=input_parameter_list, yticklabels=output_parameter_list if i==0 else False, cbar=i==(len(sa_results)-1), cbar_kws={'label': sa_sobol_indice})
                axes.set_title(model)   
            else:
                sns.heatmap(np.array(sensitivity_values), cmap="viridis", ax=axes[i], xticklabels=input_parameter_list, yticklabels=output_parameter_list if i==0 else False, cbar=i==(len(sa_results)-1), cbar_kws={'label': sa_sobol_indice})            
                axes[i].set_title(model)            

    plt.tight_layout()
    save_plot(plt, output_path + title)

def plot_sa_results_17_segments(sa_results_dict, input_names, output_path, sa_17_segment_model, sa_sobol_indice):
    """Plots and saves the 17-segments plot.
    Args:
        - sa_results_dict: dataFrame -> contains sensitivity analysis results
        - input_names: list -> list of str with names of inputs
        - output_path: str -> path to save the plot
        - sa_17_segment_model: Surrogate Model -> which model shall be used
        - sa_sobol_indice: str -> Sobol indice for sa (ST or S1)
    """
    fig, axes = plt.subplots(1, len(sa_results_dict[sa_17_segment_model]['TVPG'][sa_sobol_indice]), figsize=(6.5, 2))
    radii = np.array([1, 2, 3, 4])
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=1.0)#vmax=max(segment_data.values()))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    num_segments = [1, 4, 6, 6]
    start_angles = [0, np.pi/4, 0, 0]
    segment_annotation = [17, 13, 14, 15, 16, 12, 7, 8, 9, 10, 11, 6, 1, 2, 3, 4, 5]
    segment_data = {}
    for k, input_name in enumerate(input_names):
        for (output_name, output_dict) in sa_results_dict[sa_17_segment_model].items():
            segment_data[output_name] = output_dict[sa_sobol_indice][k]
        segment_count = 0

        # for each ring in the plot
        for i, radius in enumerate(radii):      

            # for each segment in a ring
            for j in range(num_segments[i]):
                segment_count = segment_count + 1
                angle = 2 * np.pi / num_segments[i]
                start_angle = start_angles[i] + j * angle
                end_angle = start_angles[i] + (j+1) * angle
                segment = segment_annotation[segment_count-1]
                value = segment_data['wss-{}'.format(segment)]
                color = cmap(value)
                wedge = patches.Wedge((0, 0), radius, start_angle * 180 / np.pi, end_angle * 180 / np.pi, width=1, color=color)
                axes[k].add_patch(wedge)

                # Text in each wedge
                text_color = 'white' if value < 0.5 else 'black'  # Farbe des Textes basierend auf dem Wert
                center_angle = (start_angle + end_angle) / 2
                center_radius = (radii[i-1]+radii[i]) / 2
                x = 0 if segment_count == 1 else center_radius * np.cos(center_angle)
                y = 0 if segment_count == 1 else center_radius * np.sin(center_angle)
                axes[k].text(x, y, segment_annotation[segment_count-1], ha='center', va='center', color=text_color)

        # axes configurations
        axes[k].axis('off')
        lim = 4.2
        axes[k].set_xlim(-lim, lim)
        axes[k].set_ylim(-lim, lim)
        axes[k].set_aspect('equal')
        axes[k].set_title(input_dict[input_name])
        if k == len(radii)-1:
            cbar = plt.colorbar(sm, ax=axes[k], orientation ='vertical', fraction=0.05)
            cbar.set_label(sa_sobol_indice)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_useOffset(False)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.tight_layout()    
    save_plot(plt,  output_path + sa_sobol_indice+" 17 Segments")

""" ----- For Uncertainty Quantification ----- """
def bounds_mean_std(data, output_path, output_parameters=[0], output_names = ['default'], model='GP', is_title=True, title="Bounds Variation plot", \
                    x_annot="Input Variation", y_annot="Output Variation", all_in_one=False, annotation='all', figsize = (6.5, 9.0)):
    uncertainties = sorted(data.keys())
    num_params = len(output_parameters)
    num_plots = len(data[uncertainties[0]][model][0])

    fig, axes = plt.subplots(num_params if all_in_one == False else 1, 1, figsize=figsize)
    cm = plt.get_cmap('gist_rainbow')
    for i, output_parameter in enumerate(output_parameters):
        plot_data = []
        for uncertainty in uncertainties:
            data_u_m_o = data[uncertainty][model][:, output_parameter]
            plot_data.append(data_u_m_o[~np.isnan(data_u_m_o)])

        mean_values = np.array([np.mean(data) for data in plot_data])
        std_values = np.array([np.std(data) for data in plot_data])

        print("std_val: ", output_names[i], " ",std_values)
        print("std_prc: ", output_names[i], " ",100*std_values/mean_values)

        # Plotting
        ax = axes[i] if num_params > 1 and all_in_one == False else axes
        ax.plot(uncertainties, mean_values, color=cm(i//1 * 1.0/num_plots), label=output_dict[output_names[i]]+" Mean")
        ax.fill_between(uncertainties, mean_values - std_values, mean_values + std_values, color=cm(i//1 * 1.0/num_plots), alpha=0.4, label='SD' if num_params==1 else None)
        ax.set_xlabel(x_annot)
        ax.set_ylabel(y_annot)
        if all_in_one==False:
            ax.set_title(output_names[i])#f'Parameter {output_parameter}')            
        elif annotation != 'legend' and not thesis:
            if num_params <4: ax.legend()
            else: ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
        for j, (x, mean_val, std_val) in enumerate(zip(uncertainties, mean_values, std_values)):
            mean_val
            std = r'$SD_p$'
            if annotation == 'all': ax.text(x, mean_val, f'Mean={format_value(mean_val)}\nStd={format_value(std_val)}\n{std}={100*std_val/mean_val:.2f}%', ha='center', va='bottom')
            if annotation == 'pstd' and j == len(uncertainties)-1: ax.text(x, mean_val, f'{std}={100*std_val/mean_val:.2f}%', ha='center', va='bottom')
            if annotation == 'legend' and j == len(uncertainties)-1: ax.text(x, mean_val, f'{output_dict[output_names[i]]}, {std}={100*std_val/mean_val:.2f}%', ha='center', va='bottom')

    # Title and other adjustments
    fig.suptitle(title if is_title else None)
    plt.tight_layout()
    save_plot(plt, output_path + title + "_" + model + "_" + annotation)

def bounds_sobol(data, output_path, input_labels, output_labels, model_name='GP', sobol_index='ST', 
                 fig_size=(6.5, 9), font_size=10, is_title=True, title="Bounds sobol", x_annot="Input Variation", y_annot="Sensitivity"):    

    plt.ticklabel_format(axis='y', style='sci', scilimits= (0,0))

    uncertainty_values_to_plot = list(data.keys())  # Uncertainty values to plot
    #output_names = list(data[uncertainty_values_to_plot[0]][model_name].keys())
    output_names = output_labels.keys()
    
    if len(output_names) < 3:
        row = 1
        col = len(output_names)
    else:
        row = col = math.ceil(math.sqrt(len(output_names)))

    fig, axes = plt.subplots(row,col, figsize=fig_size)
    plt.subplots_adjust(bottom=0.5, 
                    left=0.1, 
                    top = 0.975, 
                    right=0.975, hspace=0.7)

    i=j=k=0
    for _, output in enumerate(output_names):
        ax = axes[j][i] if row > 2 else axes[i]

        # Extract data to plot
        sobol_values = []
        for uncertainty_value in uncertainty_values_to_plot:
            sobol_values.append(data[uncertainty_value][model_name][output][sobol_index][:])

        # Plotting
        ax.plot(uncertainty_values_to_plot, sobol_values, marker='o', markersize = 3, linewidth=2)
        ax.set_title(output_labels[output], fontsize=font_size)
        if i == 0: ax.set_ylabel(f"S$_{{{sobol_index[-1]}}}$")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x)))
        if len(output_names) - col - 1 < k < len(output_names): ax.set_xlabel(x_annot)
        #if round((i-1)/num_cols) == num_rows-1: ax.set_xlabel(" ")
        ax.grid(True)
        if i == col - 1:
            j += 1
            i = 0
        else:
            i += 1
        k+=1     
    
    # Delete the remaining empty axises
    if i != col - 1: 
        while j <= row - 1:
            while i <= row - 1:
                axes[j][i].set_axis_off()
                i += 1 
            j += 1
            i = 0    
    fig.legend(labels=input_labels, bbox_to_anchor=(1.01, 0.5), loc='center left')
    #fig.legend(labels=input_labels, ncol=len(input_labels))
    plt.suptitle(title if is_title else None, fontsize=font_size)
    plt.tight_layout()
    save_plot(plt, output_path + title + '_' + model_name + '_' + sobol_index)
    plt.show()

def bounds_variation_plot(data, output_parameter, output_labels, output_path, is_title=True, title="Bounds Variation plot", x_annot="Input Variation", y_annot="Output Variation", legend=True):
    """Plots and saves the distribution of each feature of the DataFrame.
    Args:
        - X_dict: dict -> Dictionary containing arrays for each model's test results
        - Y_dict: dict -> Dictionary containing labels for each model
        - output_path: str -> path to save the plot
        - is_title: bool -> Whether to display the title
        - title: str -> Title of the plot / name of the saved figure
    """
    uncertainties = sorted(data.keys())
    param_names = output_parameter

    #num_plots = len(data[uncertainties[0]])
    num_plots = len(output_parameter)

    # Custom color palette
    cm = plt.get_cmap('gist_rainbow')

    plt.ticklabel_format(axis='y', style='sci', scilimits= (0,0))
    _, ax = plt.subplots(figsize=(3.5,4.5))

    for i in range(num_plots, 0,-1):
        output_variation = []
        for uncertainty in uncertainties:
            output_variation.append(data[uncertainty][i-1])
        if output_labels:
            line = ax.plot(np.array(uncertainties), output_variation, label=output_labels[i-1], linewidth = 2. if i-1 < 4 else 1.)
        else:
            line = ax.plot(np.array(uncertainties), output_variation, linewidth = 2. if i-1 < 4 else 1.)
        line[0].set_color(cm(i//1 * 1.0/num_plots))
        
    ax.set_xlabel(x_annot)
    ax.set_ylabel(y_annot)
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(right=True)

    if legend: ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small')

    plt.title(title if is_title else None)
    plt.tight_layout()
    save_plot(plt, output_path + title)

""" ----- Additional Functions ----- """
def show_plots():
    """Shows plots"""
    plt.show()

def save_plot(plt, file_path, dpi=None):
    """Saves a plot.
    Args:
        - plt: plt -> the specific plot
        - file_path: str -> path where to save the plot
    """
    # Check directory and creating directory if necessary
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save figures
    print("   Plots being saved as type: ", types)
    for type in types:
        if dpi:
            plt.savefig(file_path + '.' + type, format=type, bbox_inches='tight', dpi=dpi)
        else:
            plt.savefig(file_path + '.' + type, format=type, bbox_inches='tight')

def get_figsize():
    """Returns the standard figure size
    
    Returns:
        Tupel -> standard figsize
    """
    return (6.5, 4)

def format_value(value):
    if abs(value) < 1e-2 or abs(value) > 1e3:
        return f"{value:.2e}"  # Use scientific notation
    else:
        return f"{value:.2f}"  # Use fixed-point notation with 2 decimal places

# Print functions
def print_column_stats(df):
    """Prints important statistics about each column in a DataFrame.
    Args:
    - df: DataFrame -> contains data
    """
    column_stats = df.describe().transpose()

    for col in df.columns:
        min_val = column_stats.loc[col, 'min']
        max_val = column_stats.loc[col, 'max']
        variation = 100*(max_val-min_val)/min_val

        print(f"Column: {col}")
        print(f"Minimum: {min_val:}")
        print(f"Maximum: {max_val:}")
        print(f"Variation (%): {variation:.3f}%")
        # Add other relevant information as needed

        print("\n")

def print_dict_stats(dict, model, id='none'):
    model_return = model
    for i, (model, values) in enumerate(dict.items()):
        #print("\n")
        #print("Model: ", model)
        Q1 = np.percentile(values, 25, axis=0)
        Q3 = np.percentile(values, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_mask = (values < lower_bound) | (values > upper_bound)
        values[outliers_mask] = np.nan  # Mark outliers as NaN
        min_values = np.nanmin(values, axis=0)
        max_values = np.nanmax(values, axis=0)
        variation = 100*(max_values-min_values)/min_values
        if model == model_return or i==0: variation_return = variation
    return variation_return
