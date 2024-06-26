#!/bin/bash
#PBS -l select=1:ncpus=1:mem=50GB
#PBS -l walltime=20:00:00
#PBS -l software=python
#PBS -k oe
#PBS -q allq

source /etc/profile.d/modules.sh
module --silent load python/3.8.3

cd $PBS_O_WORKDIR

source ./envs/rjhmc-env/bin/activate

# # HMC-DFI Inner Kernel Method
# for i in $(seq 1 1 10)
# do
#    python ./src/rjhmc-tree.py --dataset iris --datapath ./datasets/iris/ --out_dir ./results/iris/ --h_init 0.01 --h_final 0.01 --alpha 0.7 --beta 1.0 --N 800 --N_warmup 200 --save 1 --plot_info 0 --tag dfi-${i} --init_id ${i}  --method 1
#     python ./src/rjhmc-tree.py --dataset cgm --datapath ./datasets/cgm/ --out_dir ./results/cgm/ --h_init 0.01 --h_final 0.001 --alpha 0.45 --beta 2.5 --N 500 --N_warmup 500 --save 1 --plot_info 0 --tag dfi-${i} --init_id ${i}  --method 1
#    python ./src/rjhmc-tree.py --dataset breast-cancer-wisconsin --datapath ./datasets/breast-cancer-wisconsin/ --out_dir ./results/breast-cancer-wisconsin/ --h_init 0.1 --h_final 0.025 --alpha 0.95 --beta 2.0 --N 800 --N_warmup 200 --save 1 --plot_info 0 --tag dfi-${i} --init_id ${i}  --method 1
#    python ./src/rjhmc-tree.py --dataset wine --datapath ./datasets/wine/ --out_dir ./results/wine/ --h_init 0.025 --h_final 0.025 --alpha 0.7 --beta 1.5 --N 800 --N_warmup 200 --save 1 --plot_info 0 --tag dfi-${i} --init_id ${i}  --method 1
#    python ./src/rjhmc-tree.py --dataset raisin --datapath ./datasets/raisin/ --out_dir ./results/raisin/ --h_init 0.05 --h_final 0.001 --alpha 0.7 --beta 2.5 --N 800 --N_warmup 200 --save 1 --plot_info 0 --tag dfi-${i} --init_id ${i}  --method 1
# done

# # HMC-DF Inner Kernel Method
# for i in $(seq 1 1 10)
# do
# 	python ./src/rjhmc-tree.py --dataset iris --datapath ./datasets/iris/ --out_dir ./results/iris/ --h_init 0.01 --h_final 0.01  --alpha 0.45 --beta 1.0 --N 800 --N_warmup 200 --save 1 --plot_info 0 --tag ${i} --init_id ${i} --method 0
#   python ./src/rjhmc-tree.py --dataset cgm --datapath ./datasets/cgm/ --out_dir ./results/cgm/ --h_init 0.001 --h_final 0.001  --alpha 0.45 --beta 2.5 --N 500 --N_warmup 500 --save 1 --plot_info 0 --tag ${i} --init_id ${i} --method 0
#    python ./src/rjhmc-tree.py --dataset breast-cancer-wisconsin --datapath ./datasets/breast-cancer-wisconsin/ --out_dir ./results/breast-cancer-wisconsin/ --h_init 0.025 --h_final 0.025  --alpha 0.45 --beta 2.5 --N 800 --N_warmup 200 --save 1 --plot_info 0 --tag ${i} --init_id ${i} --method 0
#    python ./src/rjhmc-tree.py --dataset wine --datapath ./datasets/wine/ --out_dir ./results/wine/ --h_init 0.025 --h_final 0.025 --alpha 0.45 --beta 2.0 --N 800 --N_warmup 200 --save 1 --plot_info 0 --tag ${i} --init_id ${i} --method 0
#    python ./src/rjhmc-tree.py --dataset raisin --datapath ./datasets/raisin/ --out_dir ./results/raisin/ --h_init 0.005 --h_final 0.001 --alpha 0.45 --beta 2.0 --N 800 --N_warmup 200 --save 1 --plot_info 0 --tag ${i} --init_id ${i} --method 0
# done

exit 0
 
