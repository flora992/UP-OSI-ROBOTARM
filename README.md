# UP-OSI-ROBOTARM
COGS402 project on implementing UP-OSI on Robotic Arm Automation

## Quick Start

This section provides the minimal steps required to get started.

### Install Requirements

```bash
# TODO: Anaconda Python 3.8 or 3.9 is recommended
# TODO: Create and activate virtual env

# Download and install custom pybullet
git clone git@github.com:belinghy/bullet3.git
pip install ./bullet3

# Everything can be installed using conda or pip
# Installing packages with conda is recommended
# Below are a few examples, there are other dependencies
conda install numpy scipy matplotlib
pip install bottleneck
```

### Test Environment and Train Policies

```bash
# Test environment to see what it looks like
python run.py --mode test

# Train a policy
# trained policy is saved in directory exp/
python run.py --mode train

# Replay a policy
python run.py --mode play --net <path_to_policy_file>
```

## Interacting with PyBullet Environments

Mouse Actions: 
* Scroll to zoom, 
* Ctrl + Left click to rotate
* Other possibility: Alt + Click, Shift + Click, Right click and drag, etc

Keyboard Commands: 
* `r` to reset
* `space` to pause
* `F1` to save image
* `F2` to record video