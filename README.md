# Value Learning in Markov Decision Processes

This repository contains the source code for the paper "Learning the Value Systems of Societies with Multi-objective Preference-based Inverse Reinforcement Learning" to be submitted at AAMAS 2025. Our algorithm, SVSL-P, observes a certain MOMDP envirnment, a given set of value labels and demonstrations of preferences of a (here, simulated) society of diverse agents with different value systems (multiobjective preference weights). Then, it simultaneously learns a reward vector for the MOMDP that implements a value alignment specification for the given set of values, a set of  upto `L` preference weights that describe the different preferences observed, a clustering of agents into these value systems, and a weight-dependent policy \(\Pi(s,a|W)\) that implements any given set of weights (in particualr those selected as the clusters of the society). 

The repository includes the other algorithms used in the paper in the evaluation, namely Envelope Q-learning from MORL-baselines [MORL baselines](https://mo-gymnasium.farama.org/examples/morl_baselines/), a modification of our previous algorithm [Value Learning From Preferences](https://github.com/andresh26-uam/ValueLearningFromPreferences/tree/ECAI_VSL), and a custom implementation of [Preference-based Multi-Objective Reinforcement Learning](https://arxiv.org/html/2507.14066v1).

## Installation

## Python Environment
1. Select an empty main folder, go inside.
2. Create a virtual environment with Python 3.13+. We used 3.13.5 in the paper.
    - `python3.13 -m venv .venv`
    - `source .venv/bin/activate`
### If not changing the full code:
1. Clone the following repositories inside the main folder.
    - [MORL baselines fork](https://github.com/andresh26-uam/morl-baselines-reward.git).
    - [Mushroom RL fork](https://github.com/andresh26-uam/mushroom-rl-kz.git). Then make sure to select the branch "andres-dev".
        - `cd mushroom-rl-kz`
        - `git checkout andres-dev`
        - `cd ..`
2. Clone this repository in the main folder. 
    - `git clone https://github.com/andresh26-uam/ValueLearningInMOMDP.git`
    - `cd ValueLearningInMOMDP`
3. Install packages
    - `pip install ../mushroom-rl-kz`
    - `pip install ../morl-baselines-reward` 

4. Requirements.
    - `pip install -r full_requirements.txt`

### If planning on changing the full code (or found issues):
1. Perform the steps 1-3 from "If not chancing the full code".
1. Clone the following repositories in folder F.
    - `cd ..` (if needed to get to F)
    - Clone: [Baraacuda](https://github.com/ARAAC/baraacuda.git). Then, make sure to select the branch "andres-dev":
        - `cd baraacuda`
        - `git checkout andres-dev`
        - `cd ..`
    - Clone: [Imitation fork](https://github.com/andresh26-uam/imitationgym1.git)
4. Requirements.
    - Remove or comment lines 89 and 95 in `full_requirements.txt`.
    - `pip install -r full_requirements.txt`

## Reproduce experiments.
### Generate preference datasets (and execute the Envelope Q learning baseline)
- FF environment `sh script.sh -ffmo -genrt -algo pc -L 10 -expol envelope -pol envelope`
- MVC environment `sh script.sh -mvc -genrt -algo pc -L 10 -expol envelope -pol envelope`
### Training
The code is not memory efficient, you need at least 16GB RAM (preferably 32GB) and run only one of these simultaneously.
- SVSL-P, FF: `sh script.sh -ffmo -train -algo cpbmorl -L 10 -prefix "repr" -pdata "" -seeds 25,26,27,28,29,30,31,32,33,34`
- PbMORL, FF: `sh script.sh -ffmo -train -algo pbmorl -L 10 -prefix "repr" -pdata ""  -seeds 25,26,27,28,29,30,31,32,33,34`
- PC, FF: `sh script.sh -ffmo -train -algo pbmorl -L 10 -prefix "repr" -pdata ""  -seeds 25,26,27,28,29,30,31,32,33,34`
- SVSL-P, MVC: `sh script.sh -mvc -train -algo cpbmorl -L 15 -prefix "repr" -pdata ""  -seeds 26,27,28,29,30,31,32,33,35`
- PbMORL, MVC: `sh script.sh -mvc -train -algo pbmorl -L 15 -prefix "repr" -pdata "" -seeds 26,27,28,29,30,31,32,33,35`
- PC, MVC: `sh script.sh -mvc -train -algo pbmorl -L 15 -prefix "repr" -pdata "" -seeds 26,27,28,29,30,31,32,33,35`
