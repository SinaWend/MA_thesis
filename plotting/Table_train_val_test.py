import pandas as pd
import os
import matplotlib.pyplot as plt
from ast import literal_eval

# Define the path to the folder containing the result files
folder_path = "/home/aih/sina.wendrich/MA_thesis/zoutput_blood/benchmarks/BLOOD_acevedo_vit_different_models_2024-07-27_15-47-44"

# Define the file names
train_file = os.path.join(folder_path, "bestvalepoch_train_results.csv")
val_file = os.path.join(folder_path, "bestvalepoch_validation_results.csv")
test_file = os.path.join(folder_path, "results.csv")
# train_file = os.path.join(folder_path, "epoch5_train_results.csv")
# val_file = os.path.join(folder_path, "epoch5_validation_results.csv")
# test_file = os.path.join(folder_path, "epoch5_test_results.csv")

# Column names for reading CSV files
columns = ['param_index', 'method', 'mname', 'commit', 'algo', 'epos', 'te_d', 'seed', 'params', 'acc', 'precision', 'recall', 'specificity', 'f1', 'auroc', 'binary_precision', 'binary_recall', 'binary_specificity', 'binary_f1_score', 'acc_oracle', 'acc_val', 'model_selection_epoch', 'experiment_duration']

# Function to load data with specific options
def load_data(file_path, columns):
    return pd.read_csv(file_path, index_col=False, converters={"params": literal_eval}, skipinitialspace=True, names=columns, header=0)

# Function to split the method column
def split_method(df, column='method', parts=[1], delimiter='_'):
    # Split the method column and select specified parts
    df[column] = df[column].apply(lambda x: delimiter.join([x.split(delimiter)[i-1] for i in parts if i-1 < len(x.split(delimiter))]))
    return df

# Load the data
train_df = load_data(train_file, columns)
val_df = load_data(val_file, columns)
test_df = load_data(test_file, columns)

# Split method parts
parts_to_keep = [2]  # Change this list based on the parts you want to keep
train_df = split_method(train_df, parts=parts_to_keep)
val_df = split_method(val_df, parts=parts_to_keep)
test_df = split_method(test_df, parts=parts_to_keep)

# Define the metrics we are interested in
metrics = ['binary_recall', 'acc']

# Helper function to calculate mean and std, format them as percentage (without % sign), and return the grouped DataFrame
def calculate_stats(df, metric):
    # Calculate mean and std, group by param_index and method
    stats = df.groupby(['param_index', 'method'])[[metric]].agg(['mean', 'std'])
    # Format the mean and optionally the std into a single string as "percentage" with two decimal places, no % sign
    stats = stats[metric].apply(lambda x: f"{x['mean']*100:.2f}" if pd.isna(x['std']) else f"{x['mean']*100:.2f} ({x['std']*100:.2f})", axis=1)
    return stats

# Function to save the dataframe as an image
def save_table_as_image(df, metric, folder_path):
    num_cols = len(df.columns) + 1  # Number of columns including row labels
    num_rows = len(df) + 1  # Number of rows including column labels

    # Adjust fig size based on the metric
    if metric == 'binary_recall':
        fig_width = num_cols * 2  # Wider table for binary_recall
    else:
        fig_width = num_cols * 1.5
    fig_height = num_rows * 0.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Set frame size dynamically
    ax.axis('tight')
    ax.axis('off')
    
    table_data = df.values.tolist()
    row_labels = [idx[1] for idx in df.index]  # Use only the method for row labels
    col_labels = df.columns.tolist()
    
    # Create the table
    table = ax.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels, cellLoc='center', loc='center')
    
    # Set font size and style
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.5)  # Increase row height for better readability

    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_text_props(weight='bold', color='black')
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
    
    image_file = os.path.join(folder_path, f"{metric}_performance_summary.png")
    plt.savefig(image_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Table image for {metric} saved to {image_file}")

# Loop through each metric and generate tables and images
for metric in metrics:
    train_stats = calculate_stats(train_df, metric)
    val_stats = calculate_stats(val_df, metric)
    test_stats = calculate_stats(test_df, metric)

    # Merge the formatted stats into a single dataframe
    merged_stats = pd.concat([train_stats, val_stats, test_stats], axis=1)
    merged_stats.columns = [f'{metric}_{split}' for split in ['train', 'val', 'test']]
    
    # Save the table as an image
    save_table_as_image(merged_stats, metric, folder_path)

# Save each table as CSV files for further analysis or reporting
output_file = os.path.join(folder_path, "model_performance_summary.csv")
merged_stats.to_csv(output_file)
print(f"Summary saved to {output_file}")
