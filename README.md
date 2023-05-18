# ElliDock

### Keypoint-Free Rigid Protein Docking via Equivariant Elliptic-Paraboloid Interfaces

### Dependencies

Our code works on Linux well. You can use `environment.yml` to quickly build `conda` dependencies. Note that our environment works on `CUDA==11.4`, it will be required to install `pytorch` and other dependencies with the corresponding CUDA version on your server.

```bash
conda env create -f environment.yml
```

### DB5.5 data

The raw DB5.5 dataset was already placed in the `data` directory from the original source:

```
https://zlab.umassmed.edu/benchmark/ or https://github.com/drorlab/DIPS
```

The raw PDB files of DB5.5 dataset are in the directory `./data/benchmark5.5/structures`, with `train/val/test.txt` in the directory `./data/benchmark5.5` as the exact dataset split for use.

### DIPS data

Download the dataset (see `https://github.com/drorlab/DIPS` and `https://github.com/amorehead/DIPS-Plus`) :

```
mkdir -p ./DIPS/raw/pdb

rsync -rlpt -v -z --delete --port=33444 \
rsync.rcsb.org::ftp_data/biounit/coordinates/divided/ ./DIPS/raw/pdb
```

Follow the following first steps from `https://github.com/amorehead/DIPS-Plus` :

```bash
# Create data directories (if not already created):
mkdir project/datasets/DIPS/raw project/datasets/DIPS/raw/pdb project/datasets/DIPS/interim project/datasets/DIPS/interim/external_feats project/datasets/DIPS/final project/datasets/DIPS/final/raw project/datasets/DIPS/final/processed

# Download the raw PDB files:
rsync -rlpt -v -z --delete --port=33444 --include='*.gz' --include='*.xz' --include='*/' --exclude '*' \
rsync.rcsb.org::ftp_data/biounit/coordinates/divided/ project/datasets/DIPS/raw/pdb

# Extract the raw PDB files:
python3 project/datasets/builder/extract_raw_pdb_gz_archives.py project/datasets/DIPS/raw/pdb

# Process the raw PDB data into associated pair files:
python3 project/datasets/builder/make_dataset.py project/datasets/DIPS/raw/pdb project/datasets/DIPS/interim --num_cpus 28 --source_type rcsb --bound

# Apply additional filtering criteria:
python3 project/datasets/builder/prune_pairs.py project/datasets/DIPS/interim/pairs project/datasets/DIPS/filters project/datasets/DIPS/interim/pairs-pruned --num_cpus 28
```

The raw data `DIPS/data/DIPS/interim/pairs-pruned/` can also downloaded directly from https://www.dropbox.com/s/sqknqofy58nlosh/DIPS.zip?dl=0.

We then use the `pairs-postprocessed-*.txt` files in `DIPS/data/DIPS/interim/pairs-pruned/` for the train/valid/test sets.

### Training

The training bash file `train_db5.sh` and `train_dips.sh` are given. You can begin your training process as below:

```bash
# for DB5.5
GPU=0 bash train_db5.sh
# for DIPS
GPU=0 bash train_dips.sh
```

The only parameter you should change is `DATA_DIR` in `train_dips.sh`, which should be the storage directory of DIPS data (the path will be like `*/pairs-pruned`). You can change other hyperparameters for training in `*.sh` files.

### Inference

Test sets used in our paper are given in `test_sets_pdb/`. Ground truth (bound) structures are in `test_sets_pdb/dips_test_random_transformed/complexes/`, while unbound structures (i.e., randomly rotated and translated ligands and receptors) are in `test_sets_pdb/dips_test_random_transformed/random_transformed/` and you should precisely use those for your predictions (or at least the **receptors**, while using the ground truth ligands like we do in `test.py`).

**NOTE THAT** keep receptor or ligand position fixed and predict the transformation of the other protein will not affect the results.

The best validated model of training on DB5.5 dataset only (`DB5_best_no_finetune.ckpt`) and fine-tuned from DIPS dataset (`DB5_best_finetune.ckpt`) are provided in the directory `checkpoints`. You can inference docked receptors and test with metrics like this:

```bash
python3 test.py --dataset DB5.5 --gpu 0 --ckpt ./checkpoints/DB5_best_finetune.ckpt
```

After inference the docked receptor PDB files are saved in `checkpoints/DB5_best_finetune_results`.
