import datetime
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd


def create_run_folder(base_out="stg-synthetic-eval/outputs/experiments", config_path=None, tag=None):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}"
    if tag:
        run_name += f"_{tag}"
    run_dir = os.path.join(base_out, run_name)
    os.makedirs(run_dir, exist_ok=True)
    if config_path and os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(run_dir, 'config.yaml'))
    return run_dir, timestamp


def log_results(run_dir, results_csv, summary_png, config_path=None):
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(results_csv):
        shutil.copy(results_csv, os.path.join(run_dir, 'results.csv'))
    if os.path.exists(summary_png):
        shutil.copy(summary_png, os.path.join(run_dir, 'summary.png'))
    if config_path and os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(run_dir, 'config.yaml'))


def append_global_log(log_path, timestamp, results_csv):
    df = pd.read_csv(results_csv)
    row = {
        'timestamp': timestamp,
        'rule_f1_mean': df['rule_f1'].mean(),
        'reg_error_mean': df['reg_error'].mean(),
        'mdl_mean': df['mdl'].mean(),
    }
    pd.DataFrame([row]).to_csv(
        log_path,
        mode='a',
        index=False,
        header=not os.path.exists(log_path),
    )


def plot_progress(log_path):
    if not os.path.exists(log_path):
        return
    df = pd.read_csv(log_path)
    plt.figure(figsize=(6, 3))
    plt.plot(df['timestamp'], df['rule_f1_mean'], '-o', label='Rule-F1')
    plt.plot(df['timestamp'], df['reg_error_mean'], '-o', label='Reg-Err')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.title('Experiment Progression')
    plt.savefig(os.path.join(os.path.dirname(log_path), 'progress.png'))
    plt.close()
