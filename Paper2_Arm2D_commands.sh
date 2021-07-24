#!/usr/bin/env bash
#'''
######################Experiment 1#######################################
#'''
#'''
#Train
#'''
#'''
#Arm2D
# '''
#softlearning run_example_local examples.development --universe=gym --domain=VA --task=Energy0-v0 --exp-name=VA_E0_r1  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30
#softlearning run_example_local examples.development --universe=gym --domain=VA --task=Energy0-v0 --exp-name=VA_E0_r2  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30
#softlearning run_example_local examples.development --universe=gym --domain=VA --task=Energy0-v0 --exp-name=VA_E0_r3  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30
#softlearning run_example_local examples.development --universe=gym --domain=VA --task=Energy0-v0 --exp-name=VA_E0_r4  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30
#softlearning run_example_local examples.development --universe=gym --domain=VA --task=Energy0-v0 --exp-name=VA_E0_r5  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30

#softlearning run_example_local examples.development --universe=gym --domain=VA4dof --task=Energy0-v0 --exp-name=VA4dof_E0_r1  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30
#softlearning run_example_local examples.development --universe=gym --domain=VA4dof --task=Energy0-v0 --exp-name=VA4dof_E0_r2  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30
#softlearning run_example_local examples.development --universe=gym --domain=VA4dof --task=Energy0-v0 --exp-name=VA4dof_E0_r3  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30
#softlearning run_example_local examples.development --universe=gym --domain=VA4dof --task=Energy0-v0 --exp-name=VA4dof_E0_r4  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30
#softlearning run_example_local examples.development --universe=gym --domain=VA4dof --task=Energy0-v0 --exp-name=VA4dof_E0_r5  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30

#softlearning run_example_local examples.development --universe=gym --domain=VA6dof --task=Energy0-v0 --exp-name=VA6dof_E0_r1  --checkpoint-frequency=30   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 390
#softlearning run_example_local examples.development --universe=gym --domain=VA6dof --task=Energy0-v0 --exp-name=VA6dof_E0_r2  --checkpoint-frequency=30   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 390
#softlearning run_example_local examples.development --universe=gym --domain=VA6dof --task=Energy0-v0 --exp-name=VA6dof_E0_r3  --checkpoint-frequency=30   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 390
#softlearning run_example_local examples.development --universe=gym --domain=VA6dof --task=Energy0-v0 --exp-name=VA6dof_E0_r4  --checkpoint-frequency=30   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 390
#softlearning run_example_local examples.development --universe=gym --domain=VA6dof --task=Energy0-v0 --exp-name=VA6dof_E0_r5  --checkpoint-frequency=30   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 390

#softlearning run_example_local examples.development --universe=gym --domain=VA8dof --task=Energy0-v0 --exp-name=VA8dof_E0_r1  --checkpoint-frequency=30   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 390
#softlearning run_example_local examples.development --universe=gym --domain=VA8dof --task=Energy0-v0 --exp-name=VA8dof_E0_r2  --checkpoint-frequency=30   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 390
#softlearning run_example_local examples.development --universe=gym --domain=VA8dof --task=Energy0-v0 --exp-name=VA8dof_E0_r3  --checkpoint-frequency=30   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 390
#softlearning run_example_local examples.development --universe=gym --domain=VA8dof --task=Energy0-v0 --exp-name=VA8dof_E0_r4  --checkpoint-frequency=30   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 390
#softlearning run_example_local examples.development --universe=gym --domain=VA8dof --task=Energy0-v0 --exp-name=VA8dof_E0_r5  --checkpoint-frequency=30   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 390

#'''
#Collect action
#'''
#python examples/development/collect_actions_SAC.py --agent VA --energy Energy0-v0 --tr  _r1 --start 1 --final 30 --step 1 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA --energy Energy0-v0 --tr  _r2 --start 1 --final 30 --step 1 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA --energy Energy0-v0 --tr  _r3 --start 1 --final 30 --step 1 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA --energy Energy0-v0 --tr  _r4 --start 1 --final 30 --step 1 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA --energy Energy0-v0 --tr  _r5 --start 1 --final 30 --step 1 --gpu_choice 0

#python examples/development/collect_actions_SAC.py --agent VA4dof --energy Energy0-v0 --tr  _r1 --start 1 --final 30 --step 1 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA4dof --energy Energy0-v0 --tr  _r2 --start 1 --final 30 --step 1 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA4dof --energy Energy0-v0 --tr  _r3 --start 1 --final 30 --step 1 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA4dof --energy Energy0-v0 --tr  _r4 --start 1 --final 30 --step 1 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA4dof --energy Energy0-v0 --tr  _r5 --start 1 --final 30 --step 1 --gpu_choice 0

#python examples/development/collect_actions_SAC.py --agent VA6dof --energy Energy0-v0 --tr  _r1 --start 30 --final 390 --step 30 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA6dof --energy Energy0-v0 --tr  _r2 --start 30 --final 390 --step 30 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA6dof --energy Energy0-v0 --tr  _r3 --start 30 --final 390 --step 30 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA6dof --energy Energy0-v0 --tr  _r4 --start 30 --final 390 --step 30 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA6dof --energy Energy0-v0 --tr  _r5 --start 30 --final 390 --step 30 --gpu_choice 0

#python examples/development/collect_actions_SAC.py --agent VA8dof --energy Energy0-v0 --tr  _r1 --start 30 --final 390 --step 30 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA8dof --energy Energy0-v0 --tr  _r2 --start 30 --final 390 --step 30 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA8dof --energy Energy0-v0 --tr  _r3 --start 30 --final 390 --step 30 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA8dof --energy Energy0-v0 --tr  _r4 --start 30 --final 390 --step 30 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent VA8dof --energy Energy0-v0 --tr  _r5 --start 30 --final 390 --step 30 --gpu_choice 0

#'''
#Preprocess collected action signals
#'''
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt VA
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt VA --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt VA4dof
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt VA4dof --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt VA6dof
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt VA6dof --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt VA8dof
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt VA8dof --no_div

#python examples/plotting/AdaptiveW_process_SA.py  --agentt VA --ee  E0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt VA --ee  E0
#python examples/plotting/AdaptiveW_process_SA.py  --agentt VA4dof --ee  E0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt VA4dof --ee  E0
#python examples/plotting/AdaptiveW_process_SA.py  --agentt VA6dof --ee  E0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt VA6dof --ee  E0
#python examples/plotting/AdaptiveW_process_SA.py  --agentt VA8dof --ee  E0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt VA8dof --ee  E0

#'''
#Similar to the bar plot in the paper, compare between ASA,DSA,FSA
#'''
#python examples/plotting/compare_dof_synergy_lineplot.py Paper2_Arm2D
#python examples/plotting/compare_dof_synergy_lineplot.py Paper2_Arm2D --double_bars

#'''
#Similar to the plot in the paper, compare between P vs ASA(SEA)
#'''
#python examples/plotting/P_vs_ASA_lineplot.py Paper2_Arm2D --no_div



