# Variational Grey-Box Dynamic Matching (VGB-DM)

## Conda Environment
To create the conda environment, run the following command:

```bash
conda create -n vgb-dm python=3.12
conda activate vgb-dm
```
Then, install the required packages:
```bash
pip install -r requirements.txt     
```
build the package
```bash
pip install -e .
```

## Dataset
To generate each task dataset, please follow the instructions below.


### RLC Circuit

```bash
python src/data/generate_dataset/rlc/generate_rlc_dataset.py
```

By default it saves the dataset in `experiments/dataset/RLC`, including the training and validation/test datasets.

### Damped Pendulum
```bash
python src/data/generate_dataset/pendulum/simulate_from_conf.py --seed=1 --n-samples=1000
```
Training, validation and test can be generate by changing the seed value.

### Reaction Diffusion

```bash
python src/data/generate_dataset/reactdiff/make_dataset.py
```

To change the number of samples and seed number please edit properties (`n_samples` and `seed`) in the `src/data/generate_dataset/reactdiff/params.yaml` `yaml` file to generate different datasets.

### Lorenz System

```bash 
python src/data/generate_dataset/lorenz_attractor/generate_dataset.py  --seed=3 --n_trajectories=1000
```

## Run Models and Evaluation

To run a model training experiment, please have a look at `experiments/scripts/` where you find bash script example file run the training of the models. 

An example command to run VGB-DM on the Lorenz dataset is:

```bash
python src/runner/run_hydra.py --multirun \
    exp=lorenz \
    exp.dataset.data_path_tr="./experiments/dataset/lorenz_attractor/train/tr_lorenz_data_seed_1_f99f234a781aeda45d3ec612da565d6b.pt"\
    exp.dataset.data_path_va="./experiments/dataset/lorenz_attractor/val/val_lorenz_data_seed_2_1ec97522a1b023702f6b3e4ba4eba3b5.pt"\
    exp.algorithm=dyn-fm \
    exp.sampler.sampler_mode=pairs-history \
    exp.sampler.max_length=-1 \
    exp.sampler.enc_len_episode=30 \
    exp.enc_model.len_episode=30 \
    exp.enc_model.p_dim=2 \
    exp.enc_model.z_dim=8 \
    exp.vf_model.phys_model=true \
    exp.vf_model.vf_phys=true \
    exp.vf_model.interpolation=linear \
    exp.vf_model.history_size=4 \
    exp.training.early_stopping=false \
    exp.optimization.vf_lr=1e-3 \
    exp.optimization.enc_lr=1e-4 \
    exp.optimization.beta_p=0.1 \
    exp.optimization.beta_z=0.1 \
    exp.optimization.n_epochs=3500 \
    exp.training.max_time_tr=120 \
    exp.seed=3
```

Every best checkpoint of the running instance is saved in the `outputs/` directory, the model names comes by a hashed string of the configuration used for the training. You can find the best checkpoint in the `best_chkpt.pth` file in the corresponding model directory.

### Model Evaluation

To evaluate trained models, use the `evaluate.py` script. The evaluation script will automatically find all model checkpoints in a directory and evaluate them:

```bash
python src/evaluate/evaluate.py --root_dir=./outputs/lorenz/1cd9c8e5b1a0c7e4f2b1a3d9c8e5b1a0c7e4f2b1a3d9c8e5b1a0c7e4f2b1a3 --dataset_path=./experiments/dataset/lorenz_attractor/test/test_lorenz_data_seed_3_9c8e5b1a0c7e4f2b1a3d9c8e5b1a0c7e4f2b1a3
```
where the arguments are:
- `--root_dir`: Path to the directory containing all trained model checkpoints (searches recursively for `best_chkpt.pth` files)
- `--dataset_path`: Path to the test dataset file