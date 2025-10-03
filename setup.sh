# make virtual environment and install requirements

python3 -m venv env python=3.9
source env/bin/activate
#conda create -n env
#conda activate env

# Install required packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
#pip3 install -r requirements.txt

# Add in pythonpath to environment
REPO_PATH=$(pwd)
echo "export PYTHONPATH=$REPO_PATH" >> env/bin/activate
#~/miniconda3/envs/env/bin/activate

source env/bin/activates