# synergy_analysis

synergy_analysis is the codebase used by me during my PhD thesis.

This codebase contains codes and commands to reproduce the results in three published papers. It will be the reference for all future users who are interested to do research related to my thesis and who want to have an idea about how some results are produced in my thesis.

This implementation uses Tensorflow 2.2 and is tested under Ubuntu 18.04. 

Windows usage not supported, but Docker might be a solution for those interested. 

# Special notes
I customized the softlearning codebase to run my experiments.
Author of modification: Chai Jiazheng e-mail: jiazheng1310@gmail.com

# Getting Started

## Prerequisites

The environment can be run using conda. For conda installation, you need to have `Conda` installed. 
Also, our environments currently require a [MuJoCo](https://www.roboti.us/license.html) license.

## Mujoco Installation

1. [Download](https://www.roboti.us/index.html) and install MuJoCo 2 from the MuJoCo website. We assume that the MuJoCo files are extracted to the default location (`~/.mujoco/mujoco200`).

2. Copy your MuJoCo license key (mjkey.txt) to ~/.mujoco/mjkey.txt and ~/.mujoco/mujoco200/bin/mjkey.txt

## synergy_analysis Installation
3. Clone the codebase `synergy_analysis`

4. Create and activate conda environment, install `synergy_analysis` to enable command line interface:
```
cd ${synergy_analysis_PATH}
conda env create -f runnable.yml
conda activate tested_env
pip install -e .
```

The environment should be ready to run experiments. 

Finally, to deactivate:
```
conda deactivate
```

## To run and reproduce my results:
Please read the `synergy_analysis_tutorials.pdf` for details of the code usage.

All the essential commands are summarized in:
`essential_commands_list.sh`

To produce results of Paper1, Paper 2, and Paper 3, check:
1) `Paper1_commands.sh`
2) `Paper2_Arm2D_commands.sh`
3) `Paper2_Arm3D_commands.sh`
4) `Paper3_commands.sh`

All the commands in the files above are sequential by block, meaning each block of commands must be run before next block of commands can be run.

The experimental results are stored in `experiments_results` folder in the `synergy_analysis` codebase.

## GPU usage
While GPU does not necessary speed up the training speed since relatively simple neural networks are used in the RL framework, if you wish to use GPU, it is already functional if you have successfully created the virtual environment `tested_env` by running `conda env create -f runnable.yml`

## Troubleshooting 
It is possible that there might be some problems happening during the installation. 

If `mujoco_py` installation has some issues, make sure to follow the troubleshooting and installation guides provided on the official github page:
https://github.com/openai/mujoco-py/

If mpi4py is missing or unable to install it by pip, try:
1)    ```sudo apt-get update -y```
2)    ```sudo apt-get install -y python3-mpi4py```
3)    ```pip install mpi4py```

To solve other issues, one way is to solve the libraries version issues case by case, but make sure to follow the version of the following tricky libraries' version:
1) install serializable by: (you must uninstall it first) 
`pip install git+https://github.com/hartikainen/serializable.git@76516385a3a716ed4a2a9ad877e2d5cbcf18d4e6`
2) tensorflow==2.2.0
3) tensorflow-probability==0.10.1

# References
The codes are based on the following papers:

##### Paper 1 of my thesis:

J. Chai and M. Hayashibe, *Motor Synergy Development in High-Performing Deep
Reinforcement Learning Algorithms*, in IEEE Robotics and Automation Letters,
vol. 5, no. 2, pp. 1271-1278, April 2020.

##### Paper 2 of my thesis:

J. Chai and M. Hayashibe, *Quantification of Joint Redundancy considering Dy-
namic Feasibility using Deep Reinforcement Learning*, in ICRA 2021.

##### Paper 3 of my thesis:

J. Chai and M. Hayashibe, *Deep Reinforcement Learning with Gait Mode Specifi-
cation for Quadrupedal Trot-Gallop Energetic Analysis*, in EMBC 2021.

##### Reference to the original softlearning codebase:
If Softlearning helps you in your academic research, you are encouraged to cite their paper. Here is an example bibtex:
```
@techreport{haarnoja2018sacapps,
  title={Soft Actor-Critic Algorithms and Applications},
  author={Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine},
  journal={arXiv preprint arXiv:1812.05905},
  year={2018}
}
```
