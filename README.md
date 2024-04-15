[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

### Requirements
- Python 3.6 or later
- Conda

### Installation
1. Clone the repository
2. Create a conda environment
```bash
conda create --name drlnd python=3.12
```
3. Activate the environment
```bash
conda activate drlnd
```
4. Install the required packages
```bash
pip install -r requirements.txt
```
5. Install the requirements from the `python` folder
```bash
cd python
pip install .
```
6. Download the Unity environments from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
7. Unzip the file and place it in the root directory of the repository

### (Optional) Learning from Pixels

In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Unzip the file and place it in the root directory of the repository

### Testing the Agent

There are already pre-trained weights in the `checkpoint.pth` and `checkpoint_visual.pth` file for 2000 episodes.

The current implementation uses the environment with the ray-based perception for linux. Feel free to change it to the environment with the pixel-based perception and/or for your operating system. You can do this by changing the `env_filename` and `visual` variables in the `test.py` and `train.py` scripts.

To test the agent, run the `test.py` script.

```bash
python src/test.py
```

### Training the Agent

To train the agent, run the `train.py` script.

```bash
python src/train.py
```

## Implementation

The implementation is based on the Double Deep Q-Network (DDQN) algorithm with some modifications. The modifications include the use of a dueling network architecture and prioritized experience replay.
