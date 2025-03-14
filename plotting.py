import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import seaborn as sns
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
from sklearn.multioutput import MultiOutputRegressor
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import ticker
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
from statsmodels.multivariate.manova import MANOVA

######### THESIS SPECIFIC #########
thesis = False
units_out = ['Pa', r'Pa m$^3$', r'Pa m$^3$', 'Pa']
units_out_all = ['Pa', r'Pa m$^3$', r'Pa m$^3$', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa', 'Pa']
units_in = ['mm','mm','°','mm']
input_dict = {
    'y': r'$y_d$',
    'z': r'$z_d$',
    'alpha': r'$\alpha$',
    'R': r'$R_L$'
}
output_dict = {
    'Ekin': 'Ekin',
    'TVPG': 'TVPG',
    'Eloss': 'Eloss',
    'WSS': 'WSS',
    'wss-1': r'WSS$_1$',
    'wss-2': r'WSS$_2$',
    'wss-3': r'WSS$_3$',
    'wss-4': r'WSS$_4$',
    'wss-5': r'WSS$_5$',
    'wss-6': r'WSS$_6$',
    'wss-7': r'WSS$_7$',
    'wss-8': r'WSS$_8$',
    'wss-9': r'WSS$_9$',
    'wss-10': r'WSS$_{10}$',
    'wss-11': r'WSS$_{11}$',
    'wss-12': r'WSS$_{12}$',
    'wss-13': r'WSS$_{13}$',
    'wss-14': r'WSS$_{14}$',
    'wss-15': r'WSS$_{15}$',
    'wss-16': r'WSS$_{16}$',
    'wss-17': r'WSS$_{17}$'
}
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral' 
plt.rcParams['font.size'] = 11

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
    fig, axes = plt.subplots(1, len(metrics), figsize=figure_size)

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
            plt.legend(handles, labels, bbox_to_anchor=(1, 0.5), loc='center left')
        else:
            handles, labels = axes[0].get_legend_handles_labels()
            if thesis:
                for i, label in enumerate(labels):
                    labels[i] = output_dict[label]
            plt.legend(handles, labels, bbox_to_anchor=(1, 0.5), loc='center left', fontsize='small')
    plt.tight_layout()
    save_plot(plt, output_path + title)

def plot_densitys(X_dict, Y_dict, output_path, lables=None, is_title=True, title="Densities plot"):
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
            
        axes[i].set_xlabel(output_dict[lables[i]]+" in "+units_out_all[i])
        
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

    for i, col in enumerate(num_columns):
        axes[i].hist(df[col], bins=num_bins)

        if output_units is not None and col in output_units:
            xlabel = f"{col} in {output_units[col]}"
            axes[i].set_xlabel(xlabel, labelpad=12)
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

def plot_feature_scatterplot(df, x_cols, y_cols, output_path, is_title=True, title="Scatterplot"):
    """Plots and saves scatterplot of two columns from the DataFrame.
    
    Args:
        - df: DataFrame -> contains data
        - x_cols: list -> names of the columns for x-axis
        - y_cols: list -> names of the columns for y-axis
        - output_path: str -> path to save the plot
        - title: str -> title of plot
    """
    n_plots = len(x_cols) * len(y_cols)
    fig, axes = plt.subplots(len(y_cols), len(x_cols), figsize=(6.5, 6.5))
    for i, y_col in enumerate(y_cols):
        for j, x_col in enumerate(x_cols):
            ax = axes[i, j] if n_plots > 1 else axes
            ax.scatter(df[x_col], df[y_col], s=8)
            if thesis:
                if i == len(x_cols)-1: ax.set_xlabel(input_dict[x_col]+" in "+units_in[j])
                if j == 0: ax.set_ylabel(y_col+" in "+units_out[i])
            else:
                if i == len(x_cols)-1: ax.set_xlabel(input_dict[x_col])
                if j == 0: ax.set_ylabel(y_col)
            if is_title:
                ax.set_title(f"{title} - {input_dict[x_col]} vs {y_col}")

    plt.tight_layout()
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

def plot_data(X_df, y_df, output_path, is_title=True, title="Pairplot of Data"):
    """Plots and saves the pairplot of data.

    Args:
        - X_df: dataFrame -> contains input data
        - y_df: dataFrame -> contains output data
        - output_path: str -> path to save the plot
        - title: str -> title of plot / name of saved figure
    """
    # Create a DataFrame combining selected features and outputs
    sns.pairplot( pd.concat([X_df, y_df], axis=1))
    save_plot(plt, output_path + title)

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
            
        if is_title:
            axs[i].set_title(f'{output_name} in {output_units[output_name]}')

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

def show_plots():
    """Shows plots"""
    plt.show()

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

def plot_boxplots(X_df, y_df, output_path, is_title=True, title="Boxplots of Dataframe"):
    """Plots and saves the boxplot of a dataframe.

    Args:
        - X_df: DataFrame -> contains the data for inputs
        - y_df: DataFrame -> contains the data for boxplots
        - output_path: str -> path to save the plot
        - title: str -> title of plot and name of saved figure
    """
    num_columns = y_df.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(8,4))
    
    information = ['in ...', 'in ...', 'in ...', 'in ...','in ...', 'in ...', 'in ...', 'in ...','in ...', 'in ...', 'in ...', 'in ...','in ...', 'in ...', 'in ...', 'in ...','in ...', 'in ...', 'in ...', 'in ...','in ...', 'in ...', 'in ...', 'in ...']
    for i, column in enumerate(y_df.columns):
        ax = axes[i] if num_columns > 1 else axes
        ax.boxplot(y_df[column])
        ax.set_ylabel('Values '+information[i])
        ax.set_xticklabels([column], rotation=0)

    plt.suptitle(title if is_title else None)
    #plt.tight_layout()
    save_plot(plt, output_path + title)
    plt.show()

def plot_correlation(X_df, y_df, output_path, is_title=True, title="Correlation Matrix of Dataframe"):
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
    plt.figure(figsize=get_figsize())
    sns.heatmap(correlation_matrix, cmap='viridis', fmt=".2f")
    plt.title(title if is_title else None)
    save_plot(plt, output_path + title)

def save_plot(plt, file_path):
    """Saves a plot.

    Args:
        - plt: plt -> the specific plot
        - file_path: str -> path where to save the plot
    """
    # Check directory and creating directory if necessary
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # save figure
    plt.savefig(file_path + ".svg", format='svg', bbox_inches='tight')
    plt.savefig(file_path + ".png")
    plt.savefig(file_path + ".pdf")
    
def get_figsize():
    """Returns the standard figure size
    
    Returns:
        Tupel -> standard figsize
    """
    return (6.5, 4)

def bounds_variation_plot(data, output_parameter, output_path, is_title=True, title="Bounds Variation plot", x_annot="Input Variation", y_annot="Output Variation", legend=True):
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
    
    num_plots = len(data[uncertainties[0]])
    
    # Custom color palette
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    gray_palette = plt.cm.gray(np.linspace(0.1, 0.8, 17))  # Varying shades of gray
    custom_palette = colors + list(gray_palette)
    
    fig, ax = plt.subplots(figsize=(3.5,4.5))#get_figsize())
    
    for i in range(num_plots, 0,-1):
        output_variation = []
        for uncertainty in uncertainties:
            output_variation.append(data[uncertainty][i-1])
        ax.plot(np.array(uncertainties), output_variation, label=output_dict[param_names[i-1]], color=custom_palette[i-1], linewidth = 2. if i-1 < 4 else 1.)

    ax.set_xlabel(x_annot)
    ax.set_ylabel(y_annot)
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(right=True)
    
    if legend: ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small')
    
    plt.title(title if is_title else None)
    plt.tight_layout()
    save_plot(plt, output_path + title)

def format_value(value):
    if abs(value) < 1e-2 or abs(value) > 1e3:
        return f"{value:.2e}"  # Use scientific notation
    else:
        return f"{value:.2f}"  # Use fixed-point notation with 2 decimal places

def bounds_mean_std(data, output_path, output_parameters=[0], output_names = ['default'], model='GP', is_title=True, title="Bounds Variation plot", \
                    x_annot="Input Variation", y_annot="Output Variation", all_in_one=False, annotation='all', figsize = (6.5, 9.0)):
    uncertainties = sorted(data.keys())
    num_params = len(output_parameters)
    num_plots = len(data[uncertainties[0]][model][0])
    
    fig, axes = plt.subplots(num_params if all_in_one == False else 1, 1, figsize=figsize)
    if num_params < 10: colors = plt.get_cmap('tab10').colors
    else: colors = plt.get_cmap('tab20').colors
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
        ax.plot(uncertainties, mean_values, color=colors[i], label=output_dict[output_names[i]]+" Mean")
        ax.fill_between(uncertainties, mean_values - std_values, mean_values + std_values, color=colors[i], alpha=0.4, label='SD' if num_params==1 else None)
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

def bounds_sobol(data, output_path, output='None',model_name='GP', sobol_index='ST', inputs = ['y', 'z', 'alpha', 'R'], is_title=True, title="Bounds sobol", x_annot="Input Variation", y_annot="Sensitivity"):    
    uncertainty_values_to_plot = list(data.keys())  # Uncertainty values to plot
    outputs = list(data[uncertainty_values_to_plot[0]][model_name].keys())
    if thesis:
        for i, input in enumerate(inputs):
            inputs[i] = input_dict[input]
    num_cols = 3
    num_rows = (len(outputs) + 2) // num_cols  # Determine number of rows for subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6.5, 9))
    plt.subplots_adjust(bottom=0.1, 
                    left=0.1, 
                    top = 0.975, 
                    right=0.975, hspace=0.7)
    transformation = [3, 0, 13, 17, 18, 20, 7, 8, 9, 16, 1, 2, 11, 10, 19, 4, 12, 14, 15, 5, 6]
    outputs_transformed = [outputs[i] for i in transformation]
    for i, output in enumerate(outputs):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]

        # Extract data to plot
        sobol_values = []
        for uncertainty_value in uncertainty_values_to_plot:
            sobol_values.append(data[uncertainty_value][model_name][output][sobol_index][:])

        # Plotting
        ax.plot(uncertainty_values_to_plot, sobol_values, marker='o', markersize = 3, linewidth=2)
        ax.set_title(output_dict[output])
        if i%num_cols == 0: ax.set_ylabel(sobol_index)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x)))
        if round((i-1)/num_cols) == num_rows-1: ax.set_xlabel("Variation in mm")
        #if round((i-1)/num_cols) == num_rows-1: ax.set_xlabel(" ")
        ax.grid(True)
    
    fig.legend(labels=inputs, bbox_to_anchor=(0.5, 0.05), loc='upper center', ncol=len(inputs))
    plt.suptitle(title if is_title else None)
    #plt.tight_layout()#h_pad=0, w_pad=0)
    save_plot(plt, output_path + title + '_' + model_name + '_' + sobol_index)
    plt.show()

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

def actual_scatter_plot(X_df, y_df, output_path, is_title=True, title="Pairplot of Data reduced"):

    plot_size = max(X_df.shape[1], y_df.shape[1])
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 10
    fig, axes = plt.subplots(plot_size, plot_size, figsize=(20.,20.))
    i = 0
    j = 0

    for names, values in y_df.items():
        for names2, values2 in X_df.items():
            if j == 0:
                axes[i][j].set_ylabel(names, fontsize=30)
            if i == 6:
                if names2 == 'rpm':
                    axes[i][j].set_xlabel('ECMOrpm', fontsize= 30)
                else:
                    axes[i][j].set_xlabel(names2, fontsize=30)

            axes[i][j].scatter(X_df[names2].values, y_df[names].values, marker='.')
            j += 1
        i += 1
        j = 0

    fig.tight_layout()
    save_plot(plt, output_path + title)