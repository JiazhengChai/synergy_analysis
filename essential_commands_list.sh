#!/usr/bin/env bash

####Train#############################################################

#softlearning run_example_local examples.development --universe=gym --domain=HC --task=Energy0-v0 --exp-name=HC_E0_r1  --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC  --epoch_length 1000 --total_epoch 30

#softlearning run_example_local examples.development --universe=gym --domain=HC --task=Energy0-v0 --exp-name=HC_E0_TD3_r1  --checkpoint-frequency=100   --trial-gpus 1    --algorithm TD3  --epoch_length 1000 --total_epoch 30  --policy deterministicsPolicy

######################################################################


####Collect action########################################################
#python examples/development/collect_actions_SAC.py  --agent HC --tr  _r1 --start 100 --final 3000 --step 100

#python examples/development_TD3/collect_actions_TD3.py  --agent HC --tr   _r1 --start 100 --final 3000 --step 100

#########################################################################


######### Main commands after training and collect actions###############
# Step 1 : Extract CSV files in raw_csv folder
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --agentt HC --tr  _r11 _r12 _r13 _r14 _r15  --ee E0
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --agentt HC --tr  _v3r1 _v3r2 _v3r3 _v3r4 _v3r5  --ee E0_TD3
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --agentt HC --tr  _r11 _r12 _r13 _r14 _r15  --ee E0 --no_div
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --agentt HC --tr  _v3r1 _v3r2 _v3r3 _v3r4 _v3r5  --ee E0_TD3 --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --agentt HC4dof --tr  _r1 _r2 _r3 _r4 _r5  --ee E0
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --agentt HC4dof --tr  _r1 _r2 _r3 _r4 _r5  --ee E0 --no_div

#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --agentt HC2dof --tr  _r1 _r2 _r3 _r4 _r5  --ee E0
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --agentt HC2dof --tr  _r1 _r2 _r3 _r4 _r5  --ee E0 --no_div

# Step 2 : Process the CSV files in raw_csv folder.
#python examples/plotting/AdaptiveW_process_SA.py --agentt HC
#python examples/plotting/AdaptiveW_SA_summary.py --agentt HC
#python examples/plotting/AdaptiveW_process_SA.py --agentt HC4dof
#python examples/plotting/AdaptiveW_SA_summary.py --agentt HC4dof
#python examples/plotting/AdaptiveW_process_SA.py --agentt HC2dof
#python examples/plotting/AdaptiveW_SA_summary.py --agentt HC2dof

#Step 3 :
#Plot spatial synergy development, output folder: synergy_development_{}, eg. synergy_development_HC
#python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatial_evolution.py --agentt HC --ee E0 --tr _r11

#Plot spatiotemporal synergy development, output folder: synergy_development_{}, eg. synergy_development_HC
#python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution_SVD.py --agentt HC --ee E0 --tr _r11

#Plot spatiotemporal synergy surface area development, output folder: synergy_development_{}, eg. synergy_development_HC
#python examples/plotting/AdaptiveW_surface_area_spatiotemporal_evolution_SVD.py --agentt HC --ee E0 --tr _r11

#Plot ASA vs Performance, output folder: ASA_vs_P_line_plot
#python examples/plotting/ASA_vs_P_lineplot.py HCthree

#Plot ASA vs Performance, output folder: ASA_vs_P_inv_line_plot
#Figures similar to second paper
#python examples/plotting/P_vs_ASA_lineplot.py HCthree

#Plot performance between different agents available in process_SA_final_summary, output folder: dof_P_bar_plot or dof_P_double_bars_plot
#python examples/plotting/compare_dof_P_lineplot.py HCthree
#python examples/plotting/compare_dof_P_lineplot.py HCthree --double_bars

#Plot synergy between different agents available in process_SA_final_summary, output folder: dof_bar_plot or dof_double_bar_plot
#python examples/plotting/compare_dof_synergy_lineplot.py HCthree
#python examples/plotting/compare_dof_synergy_lineplot.py HCthree --double_bars

#Plot synergy, performance between different agents available in raw_csv (need to update in commons.py), output folder: compare_synergy_graphs
#python examples/plotting/learning_progress_compare_synergy.py HCthree

#Plot synergy, performance, performance-energy between different agents available in raw_csv (need to update in commons.py), output folder: learning_progress_graphs
# Figures similar to the first paper
#python examples/plotting/learning_progress_synergy.py

#Plot synergy metrics development throughout all learning phase, output folder: SA_evolution
#python examples/plotting/SA_variable_checkpoints.py --agentt HC

###########################################################################

