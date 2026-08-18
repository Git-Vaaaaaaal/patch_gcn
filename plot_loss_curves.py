"""
Trace les courbes de loss train/validation à partir des loss_curve_{fold}.csv
générés par utils/core_utils.py pendant l'entraînement.

Usage:
    python plot_loss_curves.py <results_dir> [--out out.png]

<results_dir> est le dossier d'un run (celui qui contient loss_curve_0.csv,
loss_curve_1.csv, ...), typiquement:
    ./results/{encoder}/5foldcv/{param_code}/{exp_code}_s{seed}/
"""

import argparse
import glob
import os
import re

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_loss_curves(results_dir):
    paths = glob.glob(os.path.join(results_dir, 'loss_curve_*.csv'))

    def fold_index(path):
        m = re.search(r'loss_curve_(\d+)\.csv$', path)
        return int(m.group(1)) if m else -1

    return sorted(paths, key=fold_index)


def plot_loss_curves(results_dir, out_path=None):
    csv_paths = find_loss_curves(results_dir)
    if not csv_paths:
        raise FileNotFoundError("Aucun loss_curve_*.csv trouvé dans {}".format(results_dir))

    n_folds = len(csv_paths)
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4), squeeze=False)
    axes = axes[0]

    for ax, csv_path in zip(axes, csv_paths):
        df = pd.read_csv(csv_path)
        fold = re.search(r'loss_curve_(\d+)\.csv$', csv_path).group(1)

        ax.plot(df['epoch'], df['train_loss'], label='train_loss')
        ax.plot(df['epoch'], df['val_loss'], label='val_loss')
        ax.set_title('Fold {}'.format(fold))
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(results_dir, 'loss_curves.png')
    fig.savefig(out_path, dpi=150)
    print("Figure sauvegardée dans {}".format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help="Dossier contenant les loss_curve_*.csv")
    parser.add_argument('--out', default=None, help="Chemin de sortie du PNG (défaut: results_dir/loss_curves.png)")
    args = parser.parse_args()

    plot_loss_curves(args.results_dir, args.out)
