[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"



# deep-reinforcement-continuous-control

This repository implements a solution to Project 2, Continuous Control, of the Udacity Deep Reinforcement Learning Nanodegree.

### How To Run

Create Conda virtual environment from requirements.txt

```
conda create --name drlnd python=3.7
conda activate drlnd
conda install --file requirements.txt
```

Then launch Jupyter Notebooks:

```
(drlnd)$ jupyter notebook
```

Inside Jupyter launch Continuous_Control.ipynb to train agent. Inside notebook run all cells from menu or manually advance through notebook with Shift+Enter. On Udacity GPU workspace this project takes about 4h to run.

See REPORT.md for explanation of approach taken. See Continous_Control.md or Continous_Control.html for log of run that solved the environment.


### Udacity Project Description

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved when the agent gets an average score of +30 (over 100 consecutive episodes, and over all agents).
