# ElliDock

### Rigid Protein-Protein Docking via Equivariant Elliptic-Paraboloid Interface Prediction

### Dependencies

Our code works on Linux well with `CUDA==11.4`. It will be required to install `pytorch` and other dependencies listed below with the corresponding CUDA version (if necessary) on your server.

```
biopython
e3nn
pandas
torch_scatter
tqdm
requests
```

### DB5.5 data

The raw DB5.5 dataset was already placed in the `data` directory from the original source:

```
https://zlab.umassmed.edu/benchmark/ or https://github.com/drorlab/DIPS
```

The raw PDB files of DB5.5 dataset are in the directory `./data/benchmark5.5/structures`, with `train/val/test.txt` in the directory `./data/benchmark5.5` as the exact dataset split for use.

### SAbDab data

We use [SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/) as the training set and [RAbD](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006112) as the test set. SAbDab is a database of antibody structures that updates on a weekly basis, and RAbD is a curated benchmark on antibody design. We have provided the summary data for SAbDab and RAbD in the summaries folder, please further download all structure data (renumbered with IMGT in PDB format) from [the downloads entry of the offical website of SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?all=true). Ensure the structure data renumbered by imgt is located at `{PDB_DIR}/all_structures/imgt`, while `PDB_DIR` is up to you to choose.

We have provided the script in `data/download_and_split.sh` to preprocess and split the data of SAbDab. Please run the following command to get your data prepared:

```bash
cd data && bash download_and_split.sh -d {PDB_DIR}
```

If all goes well, you will see `train/valid/test.json` in the `PDB_DIR` directory.

### Training

The training bash file `train_db5.sh` and `train_sabdab.sh` are given. You can begin your training process as below:

```bash
# for DB5.5
GPU=0 bash train_db5.sh
# for SAbDab
GPU=0 bash train_sabdab.sh
```

you must change `DATA_DIR` in `train_sabdab.sh`, which should be your own `PDB_DIR`. You can change other hyperparameters for training in `*.sh` files.

### Evaluation

#### TM-Score

We have provided the source cpp code `TMscore.cpp` in `evaluate`, please run:

```bash
cd evaluate && g++ -O3 -o TMscore TMscore.cpp
```

#### DockQ

We use the public tool from Github (https://github.com/bjornwallner/DockQ) to evaluate DockQ, please run:

```
cd evaluate && git clone https://github.com/bjornwallner/DockQ.git
```

### Inference

Test sets used in our paper are given in `test_sets_pdb/`. Given `dataset` in `{db5, sabdab}`, Ground truth (bound) structures are in `test_sets_pdb/{dataset}_test_random_transformed/complexes/`, while unbound structures (i.e., randomly rotated and translated ligands and receptors) are in `test_sets_pdb/{dataset}_test_random_transformed/random_transformed/` and you should precisely use those for your predictions.

The best validated model on DB5.5 (`db5_best.ckpt`) and the best validated model on SAbDab (`sabdab_best.ckpt`) have been stored in `checkpoints`. You can inference docked complexes and obtain the same metrics listed in our paper with the following command:

```bash
python test.py --dataset DB5 --gpu {GPU} --ckpt ./checkpoints/db5_best.ckpt
```

After the inference step, the docked complex PDB files are saved in `checkpoints/db5_best_results`.

To obtain metrics of other methods, please run:

```bash
python test.py --model_type {HDock/Multimer/DiffDock-PP/EquiDock} --dataset {DB5/SAbDab} --gpu {GPU}
```

If you only want to generate the docked complex structure without evaluation, use the following command:

```bash
python inference.py --dataset {DB5/SabDab} --gpu {GPU} --ckpt {ckpt_path}
```

### License
MIT
