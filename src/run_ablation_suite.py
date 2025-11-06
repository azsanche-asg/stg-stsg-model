import argparse
import datetime
import os
import shutil
import subprocess

from src.experiment_tracker import create_run_folder


def run_cmd(cmd, cwd=None, env=None):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, check=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='Path to synthetic facade images')
    parser.add_argument('--base_dataset', required=True, help='Base dataset path under stg-synthetic-eval/datasets')
    parser.add_argument('--config', required=True, help='Path to v1_facades.yaml')
    args = parser.parse_args()

    experiments = [
        {'name': 'v1_baseline', 'flags': []},
        {'name': 'noRepeat', 'flags': ['--no_repeat']},
        {'name': 'noSplit', 'flags': ['--no_split']},
        {'name': 'noRenderer', 'flags': ['--no_renderer']},
    ]

    gt_base = os.path.join(args.base_dataset)
    for exp in experiments:
        name = exp['name']
        flags = exp['flags']
        dataset_out = os.path.join(args.base_dataset + '_' + name)
        os.makedirs(dataset_out, exist_ok=True)

        gt_files = [fname for fname in os.listdir(gt_base) if fname.endswith('_gt.json')]
        for fname in gt_files:
            src = os.path.join(gt_base, fname)
            dst = os.path.join(dataset_out, fname)
            if not os.path.exists(dst):
                try:
                    shutil.copy(src, dst)
                except Exception as exc:
                    print(f'[warn] could not copy {src} -> {dst}: {exc}')

        run_dir, timestamp = create_run_folder(tag=name, config_path=args.config)
        print(f"\n=== Running experiment: {name} ===")
        print(f"Output folder: {dataset_out}\n")
        print(f"Experiment log folder: {run_dir}\n")

        cmd_infer = [
            'python', '-m', 'src.infer_v1',
            '--images', args.images,
            '--out', dataset_out,
            '--config', args.config,
        ] + flags
        run_cmd(cmd_infer, cwd=os.path.abspath('.'))

        print(f"\nEvaluating {name} ...")
        cfg_path = os.path.join('configs', 'synthetic_facades.yaml')
        cmd_eval = [
            'python', 'src/run_eval.py',
            '--config', cfg_path,
            '--dataset', dataset_out,
        ]
        env = os.environ.copy()
        eval_path = os.path.abspath('../stg-synthetic-eval')
        env['PYTHONPATH'] = env.get('PYTHONPATH', '') + ':' + eval_path
        env['EXP_TAG'] = name
        run_cmd(cmd_eval, cwd=eval_path, env=env)

        print(f"✅ Finished {name} at {datetime.datetime.now()}\n")

    print("\nAll experiments completed successfully. Check stg-synthetic-eval/outputs/experiments for logs.")


if __name__ == '__main__':
    main()
