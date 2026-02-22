### Preamble

Install miniconda
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
```

### Setup Instructions

```
git clone https://github.com/elizahoward/Muon_Collider_Smart_Pixels.git
cd ~/miniconda3/envs
mkdir mlgpu_qkeras_tar && tar -xzf /local/d1/smartpixML/conda_environments/mlgpu_qkeras.tar.gz -C mlgpu_qkeras_tar
conda activate mlgpu_qkeras_tar
```

Alternatively, if not working on the workstation, copy /local/d1/smartpixML/conda_environments/mlgpu_qkeras.tar.gz to your machine, and then use 
`tar -xzf /path/to/mlgpu_qkeras.tar.gz -C mlgpu_qkeras_tar` in place of the corresponding line above

### Train some models and make some plots.

```
cd MuC_Smartpix_ML
python modelRunner.py
```
