# Setup

1. Create a conda environment that will contain python 3:
```
conda create -n rl_drone python=3.9
```

2. activate the environment (do this every time you open a new terminal and want to run code):
```
source activate rl_drone
```

3. Install the requirements into this conda environment
```
cd src
pip install -r requirements.txt
```

4. Allow your code to be able to see 'acl'
```
$ pip install -e .
```

# Visualizing with Tensorboard

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```