import os
import re
import glob
import csv
import pandas as pd
from ast import literal_eval

def process_log(file_path, hyperparameters, output_dir):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract the experiment number from the file path
    experiment_info = re.search(r'run_experiment-index=(\d+)-(\d+)\.err$', file_path)
    if not experiment_info:
        return
    param_index, experiment_number = experiment_info.groups()

    # Get the method and params from hyperparameters.csv
    hyperparam_data = hyperparameters.get(param_index, {})
    method = hyperparam_data.get('method', 'unknown')
    algo = hyperparam_data.get('model', 'unknown')
    params = hyperparam_data.get('params', '{}')

    # Add quotation marks around the params value
    params = f'"{params}"'

    # File paths for output files
    train_output_path = os.path.join(output_dir, f'train_results_{param_index}-{experiment_number}.csv')
    val_output_path = os.path.join(output_dir, f'validation_results_{param_index}-{experiment_number}.csv')
    test_output_path = os.path.join(output_dir, f'test_results_{param_index}-{experiment_number}.csv')

    # Regex to find blocks of experiments
    experiment_blocks = re.split(r'(?=Experiment start at:)', content)[1:]

    # Initialize headers
    headers = "param_index, method, mname, commit, algo, epos, te_d, seed, params, acc, precision, recall, specificity, f1, auroc, binary_precision, binary_recall, binary_specificity, binary_f1_score, acc_oracle, acc_val, model_selection_epoch, experiment_duration\n"
    train_results = [headers]
    val_results = [headers]
    test_results = [headers]

    for block in experiment_blocks:
        experiment_details = re.search(r'(\d{4}md_\d{2}md_\d{2}_\d{2}_\d{2}_\d{2})_seed_(\d+)', block)
        if not experiment_details:
            continue

        mname, seed = experiment_details.groups()

        # Domain processing
        domains = {
            'training': ('Training Domain:', train_results),
            'validation': ('Validation:', val_results),
            'test': ('Test Domain \(oracle\):', test_results)
        }

        for domain, (domain_title, result_list) in domains.items():
            # Find all epochs for the current domain
            epoch_matches = list(re.finditer(rf"(epoch: (\d+)\n.*?---- {domain_title})", block, re.DOTALL))
            if not epoch_matches:
                print(f"No epochs found for {domain} in experiment {mname}_seed_{seed}")
                continue

            for match in epoch_matches:
                epoch = match.group(2)
                performance_match = re.search(rf"epoch: {epoch}\n.*?---- {domain_title}.*?scalar performance:\s*({{.*?}})", block, re.DOTALL)
                if not performance_match:
                    print(f"No performance data found for {domain} in experiment {mname}_seed_{seed} at epoch {epoch}")
                    continue

                data = eval(performance_match.group(1).replace('\'', '"'))  # Convert to valid JSON and parse
                result_line = f"{param_index}, {method}, mname_{mname}_seed_{seed}, commit_b44271c83_not_commited, {algo}, {epoch}, center0, {seed}, {params}, {data.get('acc', '')}, {data.get('precision', '')}, {data.get('recall', '')}, {data.get('specificity', '')}, {data.get('f1', '')}, {data.get('auroc', '')}, {data.get('binary_precision', '')}, {data.get('binary_recall', '')}, {data.get('binary_specificity', '')}, {data.get('binary_f1_score', '')}, , , , \n"
                
                # Print statement for debugging
                print(f"Adding line for {domain} at epoch {epoch}: {result_line}")

                result_list.append(result_line)

    # Write to files
    with open(train_output_path, 'w') as f:
        f.writelines(train_results)
    with open(val_output_path, 'w') as f:
        f.writelines(val_results)
    with open(test_output_path, 'w') as f:
        f.writelines(test_results)

# Directory containing .err files and hyperparameters.csv
log_dir_path = '/home/aih/sina.wendrich/MA_thesis/zoutput_wilds/benchmarks/CAMELYONWILD_center4_dinov2small_resnet_densenet_erm_lr1e5_bs32__3_10epochs_nofreezing_2024-07-28_19-14-12'

# Load hyperparameters from CSV
hyperparameters = {}
with open(os.path.join(log_dir_path, 'hyperparameters.csv'), 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        param_index = row['']
        hyperparameters[param_index] = {
            'method': row['method'],
            'model': row['model'],
            'params': row['params']
        }

# Process log files and save results in log_dir_path
for log_file_path in glob.glob(os.path.join(log_dir_path,'slurm_logs', 'run_experiment', '*.err')):
    process_log(log_file_path, hyperparameters, log_dir_path)

# Function to combine results
def combine_results(directory, result_type):
    combined_results = ["param_index, method, mname, commit, algo, epos, te_d, seed, params, acc, precision, recall, specificity, f1, auroc, binary_precision, binary_recall, binary_specificity, binary_f1_score, acc_oracle, acc_val, model_selection_epoch, experiment_duration\n"]
    seen_lines = set()

    # Iterate over all results files in the directory
    for file_path in glob.glob(os.path.join(directory, f'{result_type}_results_*.csv')):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            header = lines[0]  # Skip the header
            for line in lines[1:]:
                if line not in seen_lines:
                    combined_results.append(line)
                    seen_lines.add(line)

    # Write combined results to a single file
    combined_output_path = os.path.join(directory, f'combined_{result_type}_results.csv')
    with open(combined_output_path, 'w') as f:
        f.writelines(combined_results)

# Combine results for train, validation, and test
combine_results(log_dir_path, 'train')
combine_results(log_dir_path, 'validation')
combine_results(log_dir_path, 'test')

# Function to extract rows based on last epoch
def extract_last_epoch_rows(file_path):
    rows_to_include = []
    seen_lines = set()
    try:
        columns = [
            'param_index', 'method', 'mname', 'commit', 'algo', 'epos', 'te_d', 'seed',
            'params', 'acc', 'precision', 'recall', 'specificity', 'f1', 'auroc',
            'binary_precision', 'binary_recall', 'binary_specificity', 'binary_f1_score',
            'acc_oracle', 'acc_val', 'model_selection_epoch', 'experiment_duration'
        ]

        df = pd.read_csv(file_path, index_col=False, converters={"params": literal_eval}, skipinitialspace=True, names=columns, header=0)
        
        # Group by seed and get the last epoch for each seed
        last_epoch_df = df.groupby('seed').apply(lambda x: x.loc[x['epos'].idxmax()])

        for _, row in last_epoch_df.iterrows():
            line = row.to_string(header=False, index=False)
            if line not in seen_lines:
                rows_to_include.append(row)
                seen_lines.add(line)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return columns, rows_to_include

# Function to process logs and extract last epoch rows
def process_last_epoch_logs(directory):
    for result_type in ['train', 'validation', 'test']:
        combined_results = []
        header_written = False

        for file_path in glob.glob(os.path.join(directory, f'{result_type}_results_*.csv')):
            header, rows = extract_last_epoch_rows(file_path)
            if not header_written:
                combined_results.append(header)
                header_written = True
            combined_results.extend(rows)

        combined_output_path = os.path.join(directory, f'lastepoch_{result_type}_results.csv')
        combined_df = pd.DataFrame(combined_results[1:], columns=combined_results[0])
        combined_df.to_csv(combined_output_path, index=False)

# Process logs and extract last epoch data
process_last_epoch_logs(log_dir_path)
