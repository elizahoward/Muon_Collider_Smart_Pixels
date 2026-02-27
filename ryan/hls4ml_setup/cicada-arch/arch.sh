#!/bin/bash
source /etc/profile.d/conda.sh
conda activate /afs/cern.ch/user/a/aji/.conda/envs/cicada
python3 arch.py -n $1$2 -e $3 -x $4 -t $5 -p $6 -j $7 -c $8 -y $9
