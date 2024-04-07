# Install

## Requirements
- Python 3.6 or later
- Conda

## Installation
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
    - Linux Visual: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
7. Unzip the file and place it in the root directory of the repository
