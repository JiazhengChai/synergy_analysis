#!/usr/bin/env bash
#'''
######################Experiment 1#######################################
#'''
#'''
#Train
#'''
#'''
#HalfCheetah
# '''
#softlearning run_example_local examples.development --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_r1  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_r2  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_r3  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_r4  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_r5  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000

#'''
#HalfCheetahHeavy
# '''
#softlearning run_example_local examples.development --universe=gym --domain=HalfCheetahHeavy --task=Energy0-v0 --exp-name=HCheavy_E0_r1  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=HalfCheetahHeavy --task=Energy0-v0 --exp-name=HCheavy_E0_r2  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=HalfCheetahHeavy --task=Energy0-v0 --exp-name=HCheavy_E0_r3  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=HalfCheetahHeavy --task=Energy0-v0 --exp-name=HCheavy_E0_r4  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=HalfCheetahHeavy --task=Energy0-v0 --exp-name=HCheavy_E0_r5  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000

#'''
#Fullcheetah
# '''
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetah --task=Energy0-v0 --exp-name=FC_E0_r1  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetah --task=Energy0-v0 --exp-name=FC_E0_r2  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetah --task=Energy0-v0 --exp-name=FC_E0_r3  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetah --task=Energy0-v0 --exp-name=FC_E0_r4  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetah --task=Energy0-v0 --exp-name=FC_E0_r5  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 3000


#'''
#Collect action
#'''
#python examples/development/collect_actions_SAC.py --agent HalfCheetah --energy Energy0-v0 --tr  _r1 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent HalfCheetah --energy Energy0-v0 --tr  _r2 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent HalfCheetah --energy Energy0-v0 --tr  _r3 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent HalfCheetah --energy Energy0-v0 --tr  _r4 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent HalfCheetah --energy Energy0-v0 --tr  _r5 --start 100 --final 3000 --step 100 --gpu_choice 0

#python examples/development/collect_actions_SAC.py --agent HalfCheetahHeavy --energy Energy0-v0 --tr  _r1 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent HalfCheetahHeavy --energy Energy0-v0 --tr  _r2 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent HalfCheetahHeavy --energy Energy0-v0 --tr  _r3 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent HalfCheetahHeavy --energy Energy0-v0 --tr  _r4 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent HalfCheetahHeavy --energy Energy0-v0 --tr  _r5 --start 100 --final 3000 --step 100 --gpu_choice 0

#python examples/development/collect_actions_SAC.py --agent FullCheetah --energy Energy0-v0 --tr  _r1 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetah --energy Energy0-v0 --tr  _r2 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetah --energy Energy0-v0 --tr  _r3 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetah --energy Energy0-v0 --tr  _r4 --start 100 --final 3000 --step 100 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetah --energy Energy0-v0 --tr  _r5 --start 100 --final 3000 --step 100 --gpu_choice 0

#'''
#Preprocess collected action signals
#'''
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt HC
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt HC --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt HCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt HCheavy --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt FC
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt FC --no_div

#python examples/plotting/AdaptiveW_process_SA.py  --agentt HC --ee  E0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt HC --ee  E0

#python examples/plotting/AdaptiveW_process_SA.py  --agentt HCheavy --ee  E0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt HCheavy --ee  E0

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FC --ee  E0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FC --ee  E0


#'''
#Plot synergy development
#'''
#python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt HC
#python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt HCheavy
#python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution_SVD.py --tr _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt FC

#'''
#Similar to the bar plot in the paper, compare between P,PI,Energy
#'''
#python examples/plotting/compare_dof_P_lineplot.py Paper1
#python examples/plotting/compare_dof_P_lineplot.py Paper1 --double_bars

#'''
#Similar to the bar plot in the paper, compare between ASA,DSA,FSA
#'''
#python examples/plotting/compare_dof_synergy_lineplot.py Paper1
#python examples/plotting/compare_dof_synergy_lineplot.py Paper1 --double_bars


#'''
#To plot learning progress, performance-energy and synergy level progresses similar to the paper, adapt learning_progress_synergy.py to your case.
#'''


#'''
######################End of Experiment 1#######################################
#'''


