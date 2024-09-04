import pandas as pd
import os
import matplotlib.pyplot as plt
from ast import literal_eval

# Define the paths to the folders containing the result files for each center
folder_paths = {
    "Center 0": "/home/aih/sina.wendrich/MA_thesis/zoutput_balanced/benchmarks/CAMELYONbalancednew_center0_dinov2small_erm_dann_dial_diva_2024-08-01_00-27-59",
    "Center 4": "/home/aih/sina.wendrich/MA_thesis/zoutput_balanced/benchmarks/CAMELYONbalancednew_center4_dinov2small_erm_dann_dial_diva_2024-08-07_17-53-28",
    # "Center 3": "/path/to/center3/results",
    # Add more centers as needed
}

# Define the file names
train_file_name = "bestvalepoch_train_results.csv"
val_file_name = "bestvalepoch_validation_results.csv"
test_file_name = "results.csv"

# Column names for reading CSV files
columns = ['param_index', 'method', 'mname', 'commit', 'algo', 'epos', 'te_d', 'seed', 'params', 'acc', 'precision', 'recall', 'specificity', 'f1', 'auroc', 'binary_precision', 'binary_recall', 'binary_specificity', 'binary_f1_score', 'acc_oracle', 'acc_val', 'model_selection_epoch', 'experiment_duration']

# Param indices to include
included_param_indices = ['0', '1', '2', '3', '4', '5', '6']
included_param_indices = ['7', '8', '9', '10', '11', '12', '13']
included_param_indices = ['14', '15', '16', '17', '18', '19', '20']
included_param_indices = ['0', '1', '2', '3', '4', '5']  # Specify which param_index values to plot, or set to None for all
#included_param_indices = ['0', '6', '7', '8', '9', '10']  # Specify which param_index values to plot, or set to None for all
#included_param_indices = ['0', '11', '12', '13', '14', '15']  # Specify which param_index values to plot, or set to None for all
#included_param_indices = ['0', '16', '17', '18', '19', '20']  # Specify which param_index values to plot, or set to None for all

# Function to load data with specific options
def load_data(file_path, columns):
    df = pd.read_csv(file_path, index_col=False, converters={"params": literal_eval}, skipinitialspace=True, names=columns, header=0)
    # Filter by included param indices
    df = df[df['param_index'].astype(str).isin(included_param_indices)]
    return df

# Function to split the method column and keep specified parts
def split_method(df, column='method', parts=[1], delimiter='_'):
    df[column] = df[column].apply(lambda x: delimiter.join([x.split(delimiter)[i-1] for i in parts if i-1 < len(x.split(delimiter))]))
    return df

# Function to combine the method and params columns into two lines
def combine_method_params(df):
    df['method_params'] = df['method'] + "\n" + df['params'].apply(lambda x: "\n".join([f"{k}={v}" for k, v in x.items()]))
    return df

# Helper function to calculate mean and std, format them, and return the grouped DataFrame
def calculate_stats(df, metric):
    stats = df.groupby(['method_params'])[[metric]].agg(['mean', 'std'])
    stats = stats[metric].apply(lambda x: f"{x['mean']*100:.2f} ({x['std']*100:.2f})", axis=1)
    return stats

# Function to save the dataframe as an image
def save_table_as_image(df, metric, folder_path, title_suffix=""):
    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    num_cols = len(df.columns) + 1
    num_rows = len(df) + 1
    fig_width = num_cols * 3  # Increase the width to make the table wider
    fig_height = num_rows * 1.8  # Further increase height to give even more space for row content

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = df.values.tolist()
    row_labels = [idx for idx in df.index]
    col_labels = df.columns.tolist()
    
    table = ax.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels, cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 4.0)  # Maximum row height scaling

    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_text_props(weight='bold', color='black')
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
    
    image_file = os.path.join(folder_path, f"{metric}_performance_summary{title_suffix}.png")
    plt.savefig(image_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Table image for {metric} saved to {image_file}")

# Function to save the dataframe as a LaTeX table
def save_table_as_latex(df, metric, folder_path, title_suffix=""):
    latex_file = os.path.join(folder_path, f"{metric}_performance_summary{title_suffix}.tex")
    df.to_latex(latex_file, index=True, multirow=True, multicolumn=True)
    print(f"LaTeX table for {metric} saved to {latex_file}")

# Loop through each center and collect the results
final_combined_stats = {}
for center, folder_path in folder_paths.items():
    # Load the data for each phase
    train_df = load_data(os.path.join(folder_path, train_file_name), columns)
    val_df = load_data(os.path.join(folder_path, val_file_name), columns)
    test_df = load_data(os.path.join(folder_path, test_file_name), columns)
    
    # Split method parts
    parts_to_keep = [1]  # Adjust based on which parts you want to keep
    train_df = split_method(train_df, parts=parts_to_keep)
    val_df = split_method(val_df, parts=parts_to_keep)
    test_df = split_method(test_df, parts=parts_to_keep)
    
    # Combine method and params into one column with two lines
    train_df = combine_method_params(train_df)
    val_df = combine_method_params(val_df)
    test_df = combine_method_params(test_df)
    
    # Calculate stats for each metric and combine them into one row per center
    for metric in ['binary_recall', 'acc']:
        train_stats = calculate_stats(train_df, metric)
        val_stats = calculate_stats(val_df, metric)
        test_stats = calculate_stats(test_df, metric)
        
        # Combine the stats into a single row, each on a new line
        combined = train_stats.to_frame(name='train')
        combined['val'] = val_stats
        combined['test'] = test_stats
        combined = combined.apply(lambda row: f"Train: {row['train']}\nVal: {row['val']}\nTest: {row['test']}", axis=1)
        
        # Store the combined results
        if metric not in final_combined_stats:
            final_combined_stats[metric] = pd.DataFrame()
        
        final_combined_stats[metric][center] = combined

# After collecting all results, transpose the data so rows are centers and columns are experiments
for metric, combined_stats in final_combined_stats.items():
    combined_stats = combined_stats.T  # Transpose to make centers rows and methods columns
    
    # Save the combined table as an image and LaTeX table in each center's folder
    for folder_path in folder_paths.values():
        # Save the full table
        save_table_as_image(combined_stats, metric, folder_path)
        save_table_as_latex(combined_stats, metric, folder_path)
        
        # Save the filtered table with param_index in the file name
        param_indices_str = "_".join(included_param_indices)
        save_table_as_image(combined_stats, metric, folder_path, title_suffix=f"_filtered_{param_indices_str}")
        save_table_as_latex(combined_stats, metric, folder_path, title_suffix=f"_filtered_{param_indices_str}")
