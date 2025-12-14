## INSTALL DEPENDENCIES

Run the following commands to install required packages:

pip install torch (version 2.0.0 or greater)

pip install torchvision (version 0.15.0 or greater)

pip install numpy (version 1.23 or greater)

pip install matplotlib (version 3.7 or greater)

The required datasets (MNIST and CIFAR-10) are automatically downloaded to ./data/ on first run.

## DATASETS USED

MNIST: handwritten digit classification (1 × 28 × 28)

CIFAR-10: natural image classification (3 × 32 × 32)

Both datasets are loaded using torchvision.datasets.

# FILES OVERVIEW

## MNIST SCRIPTS

fl_dlg_mnist.py
Runs a single federated learning round on MNIST and performs a DLG reconstruction for a selected client.
Used for generating visual reconstruction examples.

fl_dlg_mnist_experiment.py
Runs multi-round federated learning on MNIST and records:

## Global test accuracy

Reconstruction accuracy
Outputs a line plot comparing both metrics over rounds.

## CIFAR-10 SCRIPTS

fl_dlg_cifar10.py
Same as the MNIST privacy script, but adapted for CIFAR-10.
Used for single reconstruction visualizations.

fl_dlg_cifar10_experiment.py
Runs multi-round federated learning on CIFAR-10 and outputs a line plot comparing:

Global test accuracy

Reconstruction accuracy

# RUNNING THE CODE

## MNIST — SINGLE RECONSTRUCTION RUNS

Baseline (no privacy):
python fl_dlg_mnist_privacy.py

Local differential privacy:
python fl_dlg_mnist_privacy.py --local_dp --dp_noise_std 0.1

Local DP + adaptive clipping:
python fl_dlg_mnist_privacy.py --local_dp --dp_noise_std 0.1 --adaptive_clip

## CIFAR-10 — SINGLE RECONSTRUCTION RUNS

Secure aggregation + local DP + clipping:
python fl_dlg_cifar10_privacy.py --secure_agg --secure_agg_group_size 3
--local_dp --dp_noise_std 0.1 --adaptive_clip

## MULTI-ROUND ACCURACY EXPERIMENTS

These scripts generate round-by-round plots comparing training accuracy and reconstruction accuracy.

## General format:
python fl_dlg_[mnist|cifar10]_experiment.py
[--secure_agg --secure_agg_group_size N]
[--local_dp --dp_noise_std S]
[--adaptive_clip]
[--num_rounds R --num_attacks_per_round A]

## Example: MNIST baseline
python fl_dlg_mnist_experiment.py --num_rounds 10 --num_attacks_per_round 5

## Example: CIFAR-10 with all privacy mechanisms
python fl_dlg_cifar10_experiment.py --secure_agg --secure_agg_group_size 3
--local_dp --dp_noise_std 0.1 --adaptive_clip
--num_rounds 10 --num_attacks_per_round 5

Each experiment produces a line graph comparing:

Global test accuracy

Reconstruction accuracy
