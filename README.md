# RL-final-DAC-AIRL-
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

# DAC
Nowaday, the DAC algorithm can function with the expert data providede by 名洋 in slack named *size1000000_std0.01_prand0.0.pth. However, the format of expert data still need to clarify and the correlated code need to be further modified.


# Start running
## DAC
To run DAC, a simple line of code can work (make sure expert data file is .pth format):
```
python DAC.py
```

## AIRL
1. To train expert data with SAC:
```
python train_expert.py --cuda --env_id BipedalWalker-v3 --num_steps 100000 --seed 0
```
2. To collect demonstration using weight of trained expert data (the buffer_size dictates the size of demo trajectories):
```
python collect_demo.py --cuda --env_id BipedalWalker-v3 --weight weights/BipedalWalker-v3.pth --buffer_size 1000000 --std 0.01 --p_rand 0.0 --seed 0
```
3. To train IL algorithms through demonstration collected from (1.)(2 option here: GAIL and AIRL -- can be modified through algo/__init  __.py):
```
python train_imitation.py --algo airl --cuda --env_id BipedalWalker-v3 --buffer buffers/BipedalWalker-v3/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 10000 --rollout_length 2000 --seed 0
```
Here the rollout_length is the size of buffer for PPO within AIRL framework. The --buffer argument takes the collected demo from (2.) as input for train imitation learning.

## DAC-regularized AIRL
Notice that:
1. train_imitation.py and trainer.py need to be swapped to the ones in DAC-regularized-AIRL
2. Put DAC-regularized-AIRL/dac_airl.py in gail_airl_ppo/algo

The rest is the same as how you run AIRL, except that specification of alpha is needed:
```
python train_imitation.py --algo airl --cuda --env_id BipedalWalker-v3 --buffer buffers/BipedalWalker-v3/size10000_std0.01_prand0.0.pth --num_steps 1000000 --eval_interval 10000 --rollout_length 2000 --seed 0 --alpha 0.5
```




