# RL-final-DAC-AIRL
## Setup
Use python=3.7 to download the specified version of modules in requirements.txt

(If not, serveral bugs may appear: new version has no "env(seed)", "env.step" return 5 instead of 4 values, etc)

Note:
1. If your desktop has no CUDA, change all the torch.device("cuda" if args.cuda else "cpu") to torch.device("cpu")
2. Install gym == 0.22, since its one of the few versions that simultaneously support "Hardcore" mode and seeds for "BipedalWalker-v3" environment
3. if you encounter problem about CPython, try to downgrade it to 0.29.37 and use the command
```
sudo apt-get install python3.7-dev
```
and change the _clamp function at network/policy.py to clamp


# Start running
## Dependencies

Run the following command to install the required Python modules:
```
pip install -r requirements.txt
```
Note: If you want to record through imageio, then the Python version must be 3.9+

## DAC

### 1. **Training the DAC Agent** (`train.py`):  
   Trains a DAC (TD3 + Discriminator) agent using expert demonstration data. Produces model weights that can be loaded by other scripts.

**Arguments:**
- `--num_steps`: Total number of training steps (default: `200000`)

**Example:**
```bash
python train.py --num_steps 5000
```

### 2. **Collecting Demonstration Data** (`collect_demo.py`):  
   Uses the trained DAC agent to interact with the environment and produce new demonstration data for future training or analysis.

**Arguments:**
- `--model_dir`: Directory of saved model files (default: `models`)
- `--model_name`: Model name prefix (default: `DAC_policy`)
- `--buffer_size`: Steps of data to collect (default: `50000`)
- `--std`: Action noise standard deviation (default: `0.01`)
- `--p_rand`: Probability of random action (default: `0.0`)
- `--output_demo_path`: Path for the collected demonstration data file

**Example:**
```bash
python collect_demo.py --model_dir models --model_name DAC_policy --buffer_size 50000 --std 0.01 --p_rand 0.0 --output_demo_path demo_collection/my_demo.pth
```

Output: Saves a `.pth` file with collected demonstration data.

### 3. **Recording a Video** (`record_video.py`):  
   Loads a trained DAC agent and records its behavior in a video file. Now supports a `--hardcore` flag to run the `BipedalWalkerHardcore-v3` environment.

**Arguments:**
- `--model_dir`: Directory of saved model files (default: `models`)
- `--model_name`: Model name prefix (default: `DAC_policy`)
- `--episodes`: Number of episodes to record (default: `5`)
- `--max_steps_per_episode`: Max steps per episode (default: `1500`)
- `--output_video_path`: Output video file path (default: `trained_agent_video.mp4`)
- `--hardcore`: Use `BipedalWalkerHardcore-v3` environment if set.

**Example:**
```bash
python record_video.py --model_dir models --model_name DAC_policy --episodes 2 --max_steps_per_episode 1000 --output_video_path agent_video.mp4 --hardcore
```

Output: A `agent_video.mp4` file showing the agent in the hardcore environment.

### 4. **Testing the Agent** (`testing.py`):  
   Loads a trained DAC agent and evaluates its performance across multiple test episodes. Users can specify a list of seeds, maximum steps per episode, and number of tests. Saves test results to a CSV file. Also supports a `--hardcore` flag to test in the hardcore environment.

**Arguments:**
- `--model_dir`: Directory of saved model files (default: `models`)
- `--model_name`: Model name prefix (default: `DAC_policy`)
- `--num_tests`: Number of test episodes (default: `10`)
- `--max_length`: Max steps per test episode (default: `1500`)
- `--seeds`: Comma-separated list of seeds. The number of seeds must match `--num_tests`.
- `--output_csv`: CSV file path for results (default: `test_results.csv`)
- `--hardcore`: Use `BipedalWalkerHardcore-v3` environment if set.

**Example:**
```bash
python testing.py --model_dir models --model_name DAC_policy --num_tests 3 --max_length 1000 --seeds 0,10,42 --hardcore --output_csv test_results.csv
```

Output: A `test_results.csv` file containing the returns and steps taken in each test episode using the specified seeds.


## AIRL

### 1. **Train Expert Data with SAC** (`train_expert.py`):
```
python train_expert.py --cuda --env_id BipedalWalker-v3 --num_steps 100000 --seed 0
```
**Note : Only for AIRL-only method, DAC-AIRL can skip this step**

### 2. **Collect Demonstrationa** (`collect_demo.py`):
```
python collect_demo.py --cuda --env_id BipedalWalker-v3 --weight weights/BipedalWalker-v3.pth --buffer_size 1000000 --std 0.01 --p_rand 0.0 --seed 0
```

**Note 1: Only for AIRL-only method, DAC-AIRL can skip this step**

Note 2: The buffer_size determines the size of demo trajectories

### 3. **Train AIRL** (`train_imitation.py`):
```
python train_imitation.py --algo airl --cuda --env_id BipedalWalker-v3 --buffer buffers/BipedalWalker-v3/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 10000 --rollout_length 2000 --seed 0
```

Note: Here the rollout_length is the size of buffer for PPO within AIRL framework. The --buffer argument takes the collected demo from (2.) as input for train imitation learning.

## DAC-regularized AIRL

Note that:
1. train_imitation.py and trainer.py need to be swapped to the ones in DAC-regularized-AIRL
2. Put DAC-regularized-AIRL/dac_airl.py in AIRL/algo
3. Add AIRL/algo/__init__.py with:
   ```
   from .dac_airl import AIRL_DAC
   ```
   and put the following line of code in ALGOS
   ```
   'airl+dac': AIRL_DAC
   ```
   
**The rest is the same as how you run AIRL**, except that specification of alpha is needed and use "airl+dac" as --algo argument:

```
python train_imitation.py --algo airl+dac --cuda --env_id BipedalWalker-v3 --buffer buffers/BipedalWalker-v3/size10000_std0.01_prand0.0.pth --num_steps 1000000 --eval_interval 10000 --rollout_length 2000 --seed 0 --alpha 0.5
```




