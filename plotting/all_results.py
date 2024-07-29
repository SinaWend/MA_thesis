import os
import re
import glob
import csv
import pandas as pd
from ast import literal_eval

# Function to process individual log files
def process_log(file_path, hyperparameters):
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
    
    # Get the directory of the .err file
    dir_path = os.path.dirname(file_path)
    
    # File paths for output files
    train_output_path = os.path.join(dir_path, f'train_results_{param_index}-{experiment_number}.csv')
    val_output_path = os.path.join(dir_path, f'validation_results_{param_index}-{experiment_number}.csv')
    test_output_path = os.path.join(dir_path, f'test_results_{param_index}-{experiment_number}.csv')
    
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
            epoch_matches = list(re.finditer(rf"epoch: (\d+).*?---- {domain_title}", block, re.DOTALL))
            if not epoch_matches:
                print(f"No epochs found for {domain} in experiment {mname}_seed_{seed}")
                continue

            for match in epoch_matches:
                epoch = match.group(1)
                
                # Extract performance data for each epoch
                performance_match = re.search(rf"epoch: {epoch}.*?---- {domain_title}.*?scalar performance:\s*({{.*?}})", block, re.DOTALL)
                if not performance_match:
                    print(f"No performance data found for {domain} in experiment {mname}_seed_{seed} at epoch {epoch}")
                    continue
                
                data = eval(performance_match.group(1).replace('\'', '"'))  # Convert to valid JSON and parse
                
                result_line = f"{param_index}, {method}, mname_{mname}_seed_{seed}, commit_b44271c83_not_commited, {algo}, {epoch}, center0, {seed}, {params}, {data['acc']}, {data['precision']}, {data['recall']}, {data['specificity']}, {data['f1']}, {data['auroc']}, {data['binary_precision']}, {data['binary_recall']}, {data['binary_specificity']}, {data['binary_f1_score']}, , , , \n"
                result_list.append(result_line)

    # Write to files
    with open(train_output_path, 'w') as f:
        f.writelines(train_results)
    with open(val_output_path, 'w') as f:
        f.writelines(val_results)
    with open(test_output_path, 'w') as f:
        f.writelines(test_results)

# Directory containing .err files and hyperparameters.csv
log_dir_path = '/home/aih/sina.wendrich/MA_thesis/zoutput_oversampling10/benchmarks/CAMELYON_center0_dinov2small_erm_dial_irm_lr1e5_bs16_classbalancing10_allfreeze'

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

# Process all .err files in the directory
for log_file_path in glob.glob(os.path.join(log_dir_path, '*.err')):
    process_log(log_file_path, hyperparameters)

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

# Function to read results CSV
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

# Read results from results.csv
results_csv_path = os.path.join(log_dir_path, 'results.csv')
results = read_results_csv(results_csv_path)

# Print out the results dictionary for verification
print("Results Dictionary:")
for key, value in results.items():
    print(f"{key}: {value}")

# Function to extract rows
def extract_rows(file_path, results):
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

        for _, row in df.iterrows():
            param_index = str(row['param_index']).strip()
            seed = str(row['seed']).strip()
            epoch = str(row['epos']).strip()

            line = row.to_string(header=False, index=False)
            #print(row.to_string(header=False, index=False))

            if (param_index, seed) in results and results[(param_index, seed)] == epoch and line not in seen_lines:
                rows_to_include.append(row)
                seen_lines.add(line)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return columns, rows_to_include

# Function to process logs
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
        
        # Write combined results to CSV with quotes around the params field
        with open(combined_output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(combined_results[0])
            
            for row in combined_results[1:]:
                # Ensure params field is quoted
                row = row.to_dict()
                params_value = str(row["params"]).replace('"', '')
                params_value = '"' + params_value + '"'
                
                # Remove any extra quotes
                params_value = params_value.replace('""', '"')
                
                row['params'] = params_value
                writer.writerow([row[col] for col in combined_results[0]])



# Process all result files in the directory
process_logs(log_dir_path, results)

# Function to combine results and extract last epoch data
def combine_last_epoch_results(directory, result_type):
    combined_results = ["param_index, method, mname, commit, algo, epos, te_d, seed, params, acc, precision, recall, specificity, f1, auroc, binary_precision, binary_recall, binary_specificity, binary_f1_score, acc_oracle, acc_val, model_selection_epoch, experiment_duration\n"]
    seen_lines = set()

    # Iterate over all results files in the directory
    for file_path in glob.glob(os.path.join(directory, f'{result_type}_results_*.csv')):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            header = lines[0]  # Skip the header
            last_epoch_lines = []
            current_seed = None

            for line in lines[1:]:
                columns = line.split(',')
                seed = columns[7].strip()

                # If the seed changes, save the last epoch line of the previous seed
                if current_seed is None:
                    current_seed = seed

                if seed != current_seed:
                    if last_epoch_lines[-1] not in seen_lines:
                        combined_results.append(last_epoch_lines[-1])  # Add last epoch line of the previous seed
                        seen_lines.add(last_epoch_lines[-1])
                    last_epoch_lines = []
                    current_seed = seed

                last_epoch_lines.append(line)

            # Append the last line of the last seed processed
            if last_epoch_lines and last_epoch_lines[-1] not in seen_lines:
                combined_results.append(last_epoch_lines[-1])
                seen_lines.add(last_epoch_lines[-1])

    # Write combined results to a single file
    combined_output_path = os.path.join(directory, f'lastepoch_{result_type}_results.csv')
    with open(combined_output_path, 'w') as f:
        f.writelines(combined_results)

# Combine results for train, validation, and test for last epoch
combine_last_epoch_results(log_dir_path, 'train')
combine_last_epoch_results(log_dir_path, 'validation')
combine_last_epoch_results(log_dir_path, 'test')
