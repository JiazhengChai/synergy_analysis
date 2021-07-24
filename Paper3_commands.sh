#!/usr/bin/env bash
#'''
######################Experiment 1#######################################
#'''
#'''
#Train
#'''
#'''
#No specificaiton speed 3
#'''
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=Energy0-v4 --exp-name=FCHeavy_E0_s3r1  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=Energy0-v4 --exp-name=FCHeavy_E0_s3r2  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=Energy0-v4 --exp-name=FCHeavy_E0_s3r3  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400

#'''
#Gallop gait specificaiton speed 3
#'''
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossG-v4 --exp-name=FCHeavy_SymlossG_s3r1  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossG-v4 --exp-name=FCHeavy_SymlossG_s3r2  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossG-v4 --exp-name=FCHeavy_SymlossG_s3r3  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400

#'''
#Trot gait specificaiton speed 3
#'''
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossT-v4 --exp-name=FCHeavy_SymlossT_s3r1  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossT-v4 --exp-name=FCHeavy_SymlossT_s3r2  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossT-v4 --exp-name=FCHeavy_SymlossT_s3r3  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400

#'''
#Collect action
#'''
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy Energy0-v4 --tr  _s3r1 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy Energy0-v4 --tr  _s3r2 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy Energy0-v4 --tr  _s3r3 --start 20 --final 400 --step 20 --gpu_choice 0

#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossG-v4 --tr  _s3r1 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossG-v4 --tr  _s3r2 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossG-v4 --tr  _s3r3 --start 20 --final 400 --step 20 --gpu_choice 0

#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossT-v4 --tr  _s3r1 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossT-v4 --tr  _s3r2 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossT-v4 --tr  _s3r3 --start 20 --final 400 --step 20 --gpu_choice 0

#'''
#Preprocess collected action signals
#'''
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3 --ee E0v4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3 --ee E0v4 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3 --ee sLGv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3 --ee sLGv4 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3 --ee sLTv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3 --ee sLTv4 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  E0v4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  E0v4

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLGv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLGv4

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLTv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLTv4


#'''
#Plot synergy development
#'''
#python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution_SVD.py --tr _s3r1 _s3r2 _s3r3 --ee E0v4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution_SVD.py --tr _s3r1 _s3r2 _s3r3 --ee sLGv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution_SVD.py --tr _s3r1 _s3r2 _s3r3 --ee sLTv4 --agentt FCheavy

#'''
#To plot learning progress, energy consumption, and performance-energy, see plot_performance_EMBC.py
#'''


#'''
######################End of Experiment 1#######################################
#'''

#'''
######################Experiment 2#######################################
#'''
#'''
#Train
#'''
#'''
#Gallop gait speed 5
#'''
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossG-v6 --exp-name=FCHeavy_SymlossG_s5r1  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossG-v6 --exp-name=FCHeavy_SymlossG_s5r2  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossG-v6 --exp-name=FCHeavy_SymlossG_s5r3  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400

#'''
#Trot gait  speed 5
#'''
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossT-v6 --exp-name=FCHeavy_SymlossT_s5r1  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossT-v6 --exp-name=FCHeavy_SymlossT_s5r2  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=SymlossT-v6 --exp-name=FCHeavy_SymlossT_s5r3  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400

#'''
#Collect action
#'''
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossG-v6 --tr  _s5r1 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossG-v6 --tr  _s5r2 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossG-v6 --tr  _s5r3 --start 20 --final 400 --step 20 --gpu_choice 0

#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossT-v6 --tr  _s5r1 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossT-v6 --tr  _s5r2 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy SymlossT-v6 --tr  _s5r3 --start 20 --final 400 --step 20 --gpu_choice 0

#'''
#Preprocess collected action signals
#'''
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3 --ee sLGv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s5r1 _s5r2 _s5r3  --ee sLGv6 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3  --ee sLTv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3  --ee sLTv6 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLGv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLGv6

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLTv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLTv6


#'''
#To plot learning progress, energy consumption, and performance-energy, see plot_performance_EMBC.py
#'''

#'''
######################End of Experiment 2#######################################
#'''

#'''
######################Experiment 3#######################################
#'''
#'''
#Train
#'''
#'''
#Gallop gait minSpring speed3
#'''
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=MinSpringG-v4 --exp-name=FCHeavy_MinSpringG_s3r1  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=MinSpringG-v4 --exp-name=FCHeavy_MinSpringG_s3r2  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=MinSpringG-v4 --exp-name=FCHeavy_MinSpringG_s3r3  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400

#'''
#Gallop gait maxSpring speed3
#'''
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=ExSpringG-v4 --exp-name=FCHeavy_ExSpringG_s3r1  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=ExSpringG-v4 --exp-name=FCHeavy_ExSpringG_s3r2  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400
#softlearning run_example_local examples.development --universe=gym --domain=FullCheetahHeavy --task=ExSpringG-v4 --exp-name=FCHeavy_ExSpringG_s3r3  --checkpoint-frequency=20   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 400

#'''
#Collect action
#'''
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy MinSpringG-v4 --tr  _s3r1 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy MinSpringG-v4 --tr  _s3r2 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy MinSpringG-v4 --tr  _s3r3 --start 20 --final 400 --step 20 --gpu_choice 0

#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy ExSpringG-v4 --tr  _s3r1 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy ExSpringG-v4 --tr  _s3r2 --start 20 --final 400 --step 20 --gpu_choice 0
#python examples/development/collect_actions_SAC.py --agent FullCheetahHeavy --energy ExSpringG-v4 --tr  _s3r3 --start 20 --final 400 --step 20 --gpu_choice 0

#'''
#Preprocess collected action signals
#'''
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3 --ee minSGv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3  --ee minSGv4 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3  --ee maxSGv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr __s3r1 _s3r2 _s3r3  --ee maxSGv4 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSGv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSGv4

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSGv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSGv4


#'''
#To plot learning progress, energy consumption, and performance-energy, see plot_performance_EMBC.py
#'''

#'''
######################End of Experiment 3#######################################
#'''






############### Synergy development graph ###################################
#python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution_SVD.py --tr _r11 _r12 _r13 _r14 _r15  --ee E0 --agentt HC #_r1 _r2 _r3

#python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution_SVD.py --tr _r4 _r5  --ee sL --agentt HC #_r1 _r2 _r3
#
# python examples/plotting/AdaptiveW_surface_area_spatiotemporal_evolution_SVD.py --tr _r4 _r5 --ee sL --agentt HC

############### Synergy development graph ###################################


############### FC  ###################################

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee E0v00 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _r1 _r2 _r3  --ee E0v00 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r4 _r5 _r6  --ee E0 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r4 _r5 _r6  --ee E0 --agentt FCheavy --no_div
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _sp5r1 _sp5r2 _sp5r3 --ee E0v1 --agentt FCheavy
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _sp5r1 _sp5r2 _sp5r3  --ee E0v1 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3 --ee E0v2 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3  --ee E0v2 --agentt FCheavy --no_div
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s2r1 _s2r2 _s2r3 --ee E0v3 --agentt FCheavy
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s2r1 _s2r2 _s2r3  --ee E0v3 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r4 _s3r5 _s3r6  --ee E0v4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r4 _s3r5 _s3r6   --ee E0v4 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3  --ee E0v45 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3   --ee E0v45 --agentt FCheavy --no_div
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s4r1 _s4r2 _s4r3  --ee E0v5 --agentt FCheavy
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s4r1 _s4r2 _s4r3  --ee E0v5 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r4 _s5r5 _s5r6  --ee E0v6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s5r4 _s5r5 _s5r6  --ee E0v6 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3  --ee E0v65 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s5r1 _s5r2 _s5r3  --ee E0v65 --agentt FCheavy --no_div


#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r4 _r5 _r6  --ee sLG --agentt FCheavy_sLG
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _r4 _r5 _r6   --ee sLG --agentt FCheavy_sLG --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r4 _s3r5 _s3r6  --ee sLGv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r4 _s3r5 _s3r6   --ee sLGv4 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r4 _r5 _r6  --ee sLT --agentt FCheavy_sLT
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _r4 _r5 _r6   --ee sLT --agentt FCheavy_sLT --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r4 _s3r5 _s3r6  --ee sLTv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r4 _s3r5 _s3r6   --ee sLTv4 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r4 _s5r5 _s5r6  --ee sLTv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s5r4 _s5r5 _s5r6  --ee sLTv6 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r4 _s5r5 _s5r6  --ee sLGv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s5r4 _s5r5 _s5r6  --ee sLGv6 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3  --ee lS --agentt FCheavy_lS
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3  --ee lS --agentt FCheavy_lS --no_div
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _sp5r1 _sp5r2 _sp5r3 --ee sLTv1 --agentt FCheavy
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _sp5r1 _sp5r2 _sp5r3  --ee sLTv1 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3 --ee lSv2 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3  --ee lSv2 --agentt FCheavy --no_div
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s2r1 _s2r2 _s2r3 --ee sLTv3 --agentt FCheavy
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s2r1 _s2r2 _s2r3  --ee sLTv3 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3  --ee lSv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3   --ee lSv4 --agentt FCheavy --no_div
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s4r1 _s4r2 _s4r3  --ee sLTv5 --agentt FCheavy
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s4r1 _s4r2 _s4r3  --ee sLTv5 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3  --ee lSv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s5r1 _s5r2 _s5r3  --ee lSv6 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee mS --agentt FCheavy_mS
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee mS --agentt FCheavy_mS --no_div
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _sp5r1 _sp5r2 _sp5r3 --ee sLGv1 --agentt FCheavy
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _sp5r1 _sp5r2 _sp5r3  --ee sLGv1 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3 --ee mSv2 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3  --ee mSv2 --agentt FCheavy --no_div
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s2r1 _s2r2 _s2r3 --ee sLGv3 --agentt FCheavy
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s2r1 _s2r2 _s2r3  --ee sLGv3 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3  --ee mSv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3   --ee mSv4 --agentt FCheavy --no_div
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s4r1 _s4r2 _s4r3  --ee sLGv5 --agentt FCheavy
##python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s4r1 _s4r2 _s4r3  --ee sLGv5 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3  --ee mSv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s5r1 _s5r2 _s5r3  --ee mSv6 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee minSv00 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee minSv00 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r4 _r5 _r6  --ee minS --agentt FCheavy_minS
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r4 _r5 _r6 --ee minS --agentt FCheavy_minS --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3 --ee minSv2 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3  --ee minSv2 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r4 _s3r5 _s3r6  --ee minSv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r4 _s3r5 _s3r6   --ee minSv4 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3  --ee minSv45 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3   --ee minSv45 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r4 _s5r5 _s5r6   --ee minSv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s5r4 _s5r5 _s5r6   --ee minSv6 --agentt FCheavy --no_div
#
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee maxSv00 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee maxSv00 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r4 _r5 _r6 --ee maxS --agentt FCheavy_maxS
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r4 _r5 _r6  --ee maxS --agentt FCheavy_maxS --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3 --ee maxSv2 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3  --ee maxSv2 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r4 _s3r5 _s3r6   --ee maxSv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r4 _s3r5 _s3r6    --ee maxSv4 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3   --ee maxSv45 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3    --ee maxSv45 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r4 _s5r5 _s5r6   --ee maxSv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s5r4 _s5r5 _s5r6   --ee maxSv6 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3   --ee maxSv65 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s5r1 _s5r2 _s5r3   --ee maxSv65 --agentt FCheavy --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee maxSGv0 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee maxSGv0 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3 --ee maxSGv2 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3  --ee maxSGv2 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3   --ee maxSGv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3    --ee maxSGv4 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3   --ee maxSGv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3   --ee maxSGv6 --agentt FCheavy --no_div
#
#
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee minSGv0 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee minSGv0 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3 --ee minSGv2 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3  --ee minSGv2 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3   --ee minSGv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3    --ee minSGv4 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3   --ee minSGv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3   --ee minSGv6 --agentt FCheavy --no_div


#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee maxSTv0 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee maxSTv0 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3 --ee maxSTv2 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3  --ee maxSTv2 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3   --ee maxSTv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3    --ee maxSTv4 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3   --ee maxSTv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3   --ee maxSTv6 --agentt FCheavy --no_div
#
##
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee minSTv0 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _r1 _r2 _r3 --ee minSTv0 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3 --ee minSTv2 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s1r1 _s1r2 _s1r3  --ee minSTv2 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s3r1 _s3r2 _s3r3   --ee minSTv4 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _s3r1 _s3r2 _s3r3    --ee minSTv4 --agentt FCheavy --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3   --ee minSTv6 --agentt FCheavy
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr _s5r1 _s5r2 _s5r3   --ee minSTv6 --agentt FCheavy --no_div

#
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy_sLT
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy_sLT
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy_sLG
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy_sLG
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLGv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLGv4
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLTv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLTv4
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLGv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLGv6
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLTv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLTv6

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy
##python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  E0v1
##python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  E0v1
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  E0v2
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  E0v2
##python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  E0v3
##python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  E0v3
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  E0v4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  E0v4
##python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  E0v5
##python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  E0v5
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  E0v6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  E0v6
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  E0v00
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  E0v00
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  E0v45
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  E0v45
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  E0v65
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  E0v65

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy_lS
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy_lS
##python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLGv1
##python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLGv1
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  lSv2
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  lSv2
##python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLGv3
##python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLGv3
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  lSv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  lSv4
##python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLGv5
##python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLGv5
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  lSv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  lSv6
#
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy_mS
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy_mS
##python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLTv1
##python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLTv1
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  mSv2
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  mSv2
##python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLTv3
##python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLTv3
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  mSv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  mSv4
##python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  sLTv5
##python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  sLTv5
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  mSv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  mSv6

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy_minS
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy_minS
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSv2
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSv2
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSv4
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSv6
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSv00
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSv00
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSv45
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSv45


#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy_maxS
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy_maxS
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSv2
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSv2
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSv4
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSv6
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSv00
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSv00
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSv45
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSv45
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSv65
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSv65

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSGv0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSGv0
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSGv2
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSGv2
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSGv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSGv4
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSGv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSGv6

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSTv0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSTv0
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSTv2
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSTv2
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSTv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSTv4
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  maxSTv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  maxSTv6

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSGv0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSGv0
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSGv2
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSGv2
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSGv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSGv4
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSGv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSGv6

#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSTv0
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSTv0
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSTv2
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSTv2
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSTv4
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSTv4
#python examples/plotting/AdaptiveW_process_SA.py  --agentt FCheavy --ee  minSTv6
#python examples/plotting/AdaptiveW_SA_summary.py  --agentt FCheavy --ee  minSTv6

#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_maxSG --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_maxSG
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_minSG --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_minSG
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_maxST --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_maxST
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_minST --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_minST
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_defS --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_defS
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_defmaxS --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_defmaxS
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_defminS --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpeed_defminS

#python examples/plotting/learning_progress_compare_synergy.py VarySpring_v4 --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_v4
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_Gv4 --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_Gv4
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_Tv4 --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_Tv4
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_UNL --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_UNL
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_GUNL --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_GUNL
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_TUNL --no_div
#python examples/plotting/learning_progress_compare_synergy.py VarySpring_TUNL

#python examples/plotting/learning_progress_compare_synergy.py VaryGait_minSv4 --no_div --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_minSv4 --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_maxSv4 --no_div --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_maxSv4 --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_minSv6 --no_div --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_minSv6 --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_maxSv6 --no_div --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_maxSv6 --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_minSUNL --no_div --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_minSUNL --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_maxSUNL --no_div --spatial_synergy
#python examples/plotting/learning_progress_compare_synergy.py VaryGait_maxSUNL --spatial_synergy

#python examples/plotting/compare_dof_P_lineplot.py FCHeavyOri  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyOri
#python examples/plotting/compare_dof_P_lineplot.py FCHeavylessSpring  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavylessSpring
#python examples/plotting/compare_dof_P_lineplot.py FCHeavymoreSpring  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavymoreSpring
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyminSpring  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyminSpring
#python examples/plotting/compare_dof_P_lineplot.py FCHeavymaxSpring  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavymaxSpring
#python examples/plotting/compare_dof_P_lineplot.py FCHeavySpeed1  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavySpeed1
#python examples/plotting/compare_dof_P_lineplot.py FCHeavySpeed3  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavySpeed3
#python examples/plotting/compare_dof_P_lineplot.py FCHeavySpeed5  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavySpeed5
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyUnlimited  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyUnlimited

#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyOri  --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyOri
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavylessSpring  --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavylessSpring
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavymoreSpring  --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavymoreSpring
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavySpeed1  --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavySpeed1
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavySpeed3  --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavySpeed3
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavySpeed5  --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavySpeed5
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyUnlimited  --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyUnlimited

#python examples/plotting/learning_progress_compare_synergy.py FCHeavyActionManip --no_div
#python examples/plotting/learning_progress_compare_synergy.py FCHeavyActionManip
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyActionManip  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyActionManip
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyActionManip --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyActionManip

#python examples/plotting/learning_progress_compare_synergy.py FCHeavyOri --no_div
#python examples/plotting/learning_progress_compare_synergy.py FCHeavyOri
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyOri  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyOri
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyOri --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyOri

#python examples/plotting/learning_progress_compare_synergy.py FCHeavyTrot --no_div
#python examples/plotting/learning_progress_compare_synergy.py FCHeavyTrot
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyTrot  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyTrot
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyTrot --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyTrot

#python examples/plotting/learning_progress_compare_synergy.py FCHeavyGallop --no_div
#python examples/plotting/learning_progress_compare_synergy.py FCHeavyGallop
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyGallop  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyGallop
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyGallop --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyGallop


#python examples/plotting/learning_progress_compare_synergy.py FCHeavySpeed5 --no_div
#python examples/plotting/learning_progress_compare_synergy.py FCHeavySpeed5
#python examples/plotting/compare_dof_P_lineplot.py FCHeavySpeed5  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavySpeed5
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavySpeed5 --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavySpeed5


#python examples/plotting/learning_progress_compare_synergy.py FCHeavyVarySpeed3 --no_div
#python examples/plotting/learning_progress_compare_synergy.py FCHeavyVarySpeed3
#python examples/plotting/learning_progress_compare_synergy.py FCHeavyVarySpeed6 --no_div
#python examples/plotting/learning_progress_compare_synergy.py FCHeavyVarySpeed6
#python examples/plotting/learning_progress_compare_synergy.py FCHeavyVarySpeed9 --no_div
#python examples/plotting/learning_progress_compare_synergy.py FCHeavyVarySpeed9
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyVarySpeed  --no_div
#python examples/plotting/compare_dof_P_lineplot.py FCHeavyVarySpeed
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyVarySpeed --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyVarySpeed



#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyGaitModeComp --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyGaitModeComp

#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyGallopTrotComp --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyGallopTrotComp

#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyGallopSpringComp --no_div
#python examples/plotting/compare_dof_synergy_lineplot.py FCHeavyGallopSpringComp


#python examples/plotting/AdaptiveW_plot_summary_histogram.py  --no_div --mylist HC_sL
#python examples/plotting/AdaptiveW_plot_summary_histogram_correlation.py  --no_div --mylist HC_sL
#python examples/plotting/AdaptiveW_plot_summary_histogram_performance.py  --no_div --mylist HC_sL


