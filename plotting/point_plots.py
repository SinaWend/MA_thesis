import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import textwrap
import os
import matplotlib.transforms as transforms

# Define the paths and filenames
file_info = [
    {'path': '/home/aih/sina.wendrich/MA_thesis/zoutput_balanced/benchmarks/CAMELYONbalancednew_center4_dinov2small_erm_dannratio_3seeds_halffreeze_2024-08-07_17-51-11', 'filename': 'results.csv', 'label': 'test_data'},
    {'path': '/home/aih/sina.wendrich/MA_thesis/zoutput_balanced/benchmarks/CAMELYONbalancednew_center4_dinov2small_erm_dannratio_3seeds_halffreeze_2024-08-07_17-51-11', 'filename': 'bestvalepoch_validation_results.csv', 'label': 'val_data'},
    {'path': '/home/aih/sina.wendrich/MA_thesis/zoutput_balanced/benchmarks/CAMELYONbalancednew_center4_dinov2small_erm_dannratio_3seeds_halffreeze_2024-08-07_17-51-11', 'filename': 'bestvalepoch_train_results.csv', 'label': 'train_data'},
]

# Read hyperparameters file (assuming all folders have the same hyperparameters.csv file)
hyperparameters_path = os.path.join(file_info[0]['path'], 'hyperparameters.csv')
hyperparameters = pd.read_csv(hyperparameters_path, index_col=0)

# Read the data from the files
dataframes = []
for info in file_info:
    df = pd.read_csv(os.path.join(info['path'], info['filename']), index_col=False, skipinitialspace=True)
    df['params'] = df['params'].apply(literal_eval)
    df['source'] = info['label']
    dataframes.append(df)

data = pd.concat(dataframes, axis=0, ignore_index=True)
data.columns = data.columns.str.strip()

# Explicitly define the category ordering for 'param_index'
param_index_order = sorted(data['param_index'].unique(), key=int)
data['param_index'] = pd.Categorical(data['param_index'], categories=param_index_order, ordered=True)
data.sort_values('param_index', inplace=True)
# Define function to clean parameter strings
def clean_params(params):
    cleaned = str(params).replace('{', '').replace('}', '').replace("'", "").strip()
    return cleaned

# Define function to truncate method string
def truncate_method(method, parts=1):
    method_parts = method.split('_')
    # Join the parts and escape underscores for proper display
    return '_'.join(method_parts[:parts]).replace('_', '\_')

# Define function to create scatter plot for specified indices and metric
def create_scatterplot(filtered_data, metric, filename_suffix, show_epoch=False, method_parts=1, ylim_0_to_1=True, add_line=False):
    num_values = len(filtered_data['param_index'].unique())
    width = max(5, 1.7 * num_values + 2)  # Adjust the width based on the number of values, with a minimum width of 8
    plt.figure(figsize=(width, 5))  # Dynamic width based on number of values

    # Determine the color palette based on the number of files
    if len(file_info) == 1:
        palette = ['black']
    else:
        palette = 'deep'

    # Plot the points
    scatter = sns.scatterplot(x='param_index', y=metric, hue='source', data=filtered_data, palette=palette)

    # Set labels
    plt.xlabel('', fontsize=20)  # Increased fontsize and removed 'Param Index' label
    plt.ylabel(metric.capitalize(), fontsize=16)  # Smaller fontsize for y-axis label

    # Adjust y-axis limits
    if ylim_0_to_1:
        plt.ylim(-0.05, 1.05)  # Set range slightly larger than 0 to 1
    else:
        plt.ylim(filtered_data[metric].min() * 0.95, filtered_data[metric].max() * 1.05)

    # Add hyperparameters information to the x-axis labels
    new_labels = []
    used_param_indices = filtered_data['param_index'].unique()
    for param_index in used_param_indices:
        if int(param_index) in hyperparameters.index:
            method = truncate_method(hyperparameters.loc[int(param_index), 'method'], parts=method_parts)
            params = clean_params(hyperparameters.loc[int(param_index), 'params'])
            if params == '':
                label = f"$\\bf{{{method}}}$"
            else:
                wrapped_params = '\n'.join(textwrap.wrap(params, width=25))  # Shorter line break width
                label = f"$\\bf{{{method}}}$\n{wrapped_params}"
            new_labels.append(label)
        else:
            new_labels.append("Unknown Parameters")
    
    ax = plt.gca()
    ax.set_xticks(range(len(new_labels)))
    ax.set_xticklabels(new_labels, rotation=-45, ha='left', fontsize=16)  # Increased fontsize and set text to left-aligned
    
    # Move the texts more to the left
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('left')
        transform = label.get_transform()
        offset = transforms.ScaledTranslation(-0.4, 0, ax.figure.dpi_scale_trans)
        label.set_transform(transform + offset)
    
    # Adjust legend to remove "source" label
    legend_labels = filtered_data['source'].unique()
    handles, _ = scatter.get_legend_handles_labels()
    plt.legend(handles, legend_labels, fontsize=10)  # Reduced fontsize to make legend smaller

    # Adjust x-axis limits to center the points when there are few values
    if num_values < 6:
        ax.set_xlim(-0.5, num_values - 0.5)
    
    # Add a horizontal line at the average value of the first element in the x-axis
    if add_line:
        first_element_value = filtered_data[filtered_data['param_index'] == used_param_indices[0]][metric].mean()
        plt.axhline(y=first_element_value, color='black', linestyle='--', linewidth=1)  # Line in black

    # Save the plot in the respective directories
    all_labels = '_'.join([info['label'] for info in file_info])
    ylim_suffix = '_ylim_0_to_1' if ylim_0_to_1 else ''
    line_suffix = '_line' if add_line else ''
    #plot_path = os.path.join(file_info[0]['path'], f"{filename_suffix}_points_{all_labels}_{metric}{ylim_suffix}{line_suffix}.png")
    plot_path = os.path.join(file_info[0]['path'], f"{filename_suffix}_{all_labels}_{metric}{ylim_suffix}{line_suffix}.pdf")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    plt.close()

# Function to filter data based on optional param_index values
def filter_data(data, param_indices=None):
    # Ensure param_index column is treated as strings for filtering
    data['param_index'] = data['param_index'].astype(str)
    
    if param_indices is not None:
        filtered = data[data['param_index'].isin(param_indices)]
    else:
        filtered = data
    
    # Sort by param_index as integers to maintain proper order
    filtered = filtered.sort_values(by='param_index', key=lambda x: x.astype(int))
    
    return filtered

# Example usage with optional param_indices
param_indices_to_plot = ['0', '1', '2', '3', '4', '5', '6']  # Specify which param_index values to plot, or set to None for all
# param_indices_to_plot = ['7', '8', '9', '10', '11', '12','13']  # Specify which param_index values to plot, or set to None for all
# param_indices_to_plot = ['14', '15', '16', '17', '18','19','20']  # Specify which param_index values to plot, or set to None for all
#param_indices_to_plot = ['0', '6', '7']  # Specify which param_index values to plot, or set to None for all
#param_indices_to_plot = ['0', '6']  # Specify which param_index values to plot, or set to None for all
#param_indices_to_plot = ['0', '1', '2', '3', '4', '5', '6','7','8','9','10']  # Specify which param_index values to plot, or set to None for all
#param_indices_to_plot = ['0', '11', '12', '13', '14', '15', '16','17','18','19','20']  # Specify which param_index values to plot, or set to None for all
#param_indices_to_plot = ['0','7','8']  # Specify which param_index values to plot, or set to None for all
#param_indices_to_plot = ['0','6', '7', '8', '9', '10', '16', '17', '18', '19', '20', '31', '32', '33', '34', '35']
# param_indices_to_plot = ['0', '1', '2', '3', '4', '5']  # Specify which param_index values to plot, or set to None for all
param_indices_to_plot = ['0', '6', '7', '8', '9', '10']  # Specify which param_index values to plot, or set to None for all
# param_indices_to_plot = ['0', '11', '12', '13', '14', '15']  # Specify which param_index values to plot, or set to None for all
# param_indices_to_plot = ['0', '16', '17', '18', '19', '20']  # Specify which param_index values to plot, or set to None for all


metrics_to_plot = ['acc', 'binary_recall']  # Specify the metrics to plot

for metric in metrics_to_plot:
    filtered_data = filter_data(data, param_indices_to_plot)
    
    # Verify the filtered data and generated labels

    # Create the plot for selected param indices
    filename_suffix_selected = f"selected_param_indices_{'_'.join(param_indices_to_plot)}" if param_indices_to_plot else "all_param_indices"
    create_scatterplot(filtered_data, metric, filename_suffix_selected, show_epoch=False, method_parts=1, ylim_0_to_1=True)
    create_scatterplot(filtered_data, metric, filename_suffix_selected, show_epoch=False, method_parts=1, ylim_0_to_1=True, add_line=True)

    # Create the plot for all param indices
    create_scatterplot(data, metric, 'all_param_indices', show_epoch=False, method_parts=1, ylim_0_to_1=True)
    create_scatterplot(data, metric, 'all_param_indices', show_epoch=False, method_parts=1, ylim_0_to_1=True, add_line=True)
