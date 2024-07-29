import os
import glob
import csv
import pandas as pd
from ast import literal_eval

def read_results_csv(file_path):
    results = {}
    try:
        # Define expected columns
        columns = [
            'param_index', 'method', 'mname', 'commit', 'algo', 'epos', 'te_d', 'seed',
            'params', 'acc', 'precision', 'recall', 'specificity', 'f1', 'auroc',
            'binary_precision', 'binary_recall', 'binary_specificity', 'binary_f1_score',
            'acc_oracle', 'acc_val', 'model_selection_epoch', 'experiment_duration'
        ]

        df = pd.read_csv(file_path, index_col=False, converters={"params": literal_eval}, skipinitialspace=True, names=columns, header=0)

        for _, row in df.iterrows():
            param_index = str(row['param_index']).strip()
            seed = str(row['seed']).strip()
            model_selection_epoch = str(row['model_selection_epoch']).strip()
            results[(param_index, seed)] = model_selection_epoch
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return results

def extract_rows(file_path, results):
    rows_to_include = []
    try:
        columns = [
            'param_index', 'method', 'mname', 'commit', 'algo', 'epos', 'te_d', 'seed',
            'params', 'acc', 'precision', 'recall', 'specificity', 'f1', 'auroc',
            'binary_precision', 'binary_recall', 'binary_specificity', 'binary_f1_score',
            'acc_oracle', 'acc_val', 'model_selection_epoch', 'experiment_duration'
        ]

        df = pd.read_csv(file_path, index_col=False, converters={"params": literal_eval}, skipinitialspace=True, names=columns, header=0)

        for _, row in df.iterrows():
            param_index = str(row['param_index']).strip()
            seed = str(row['seed']).strip()
            epoch = str(row['epos']).strip()

            if (param_index, seed) in results and results[(param_index, seed)] == epoch:
                rows_to_include.append(row)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return columns, rows_to_include

def process_logs(directory, results):
    for result_type in ['train', 'validation', 'test']:
        combined_results = []
        header_written = False

        for file_path in glob.glob(os.path.join(directory, f'{result_type}_results_*.csv')):
            header, rows = extract_rows(file_path, results)
            if not header_written:
                combined_results.append(header)
                header_written = True
            combined_results.extend(rows)

        combined_output_path = os.path.join(directory, f'bestvalepoch_{result_type}_results.csv')
        combined_df = pd.DataFrame(combined_results[1:], columns=combined_results[0])
        combined_df.to_csv(combined_output_path, index=False)

# Directory containing results files and results.csv
log_dir_path = '/home/aih/sina.wendrich/MA_thesis/zoutput_oversampling10/benchmarks/CAMELYON_center0_dinov2small_erm_dial_irm_lr1e5_bs16_classbalancing10_allfreeze'
results_csv_path = os.path.join(log_dir_path, 'results.csv')
results = read_results_csv(results_csv_path)

# Print out the results dictionary for verification
print("Results Dictionary:")
for key, value in results.items():
    print(f"{key}: {value}")

# Process all result files in the directory
process_logs(log_dir_path, results)
