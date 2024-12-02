# RL-final-DAC-AIRL-
Use python=3.7 to download the specified version of modules in requirements.txt

(If not, serveral bugs may appear: new version has no "env(seed)", "env.step" return 5 instead of 4 values, etc)

Note:
1. If your desktop has no CUDA, change all the torch.device("cuda" if args.cuda else "cpu") to torch.device("cpu")
2. 


# DAC
Nowaday, the DAC algorithm can function with the expert data providede by 名洋 in slack named *size1000000_std0.01_prand0.0.pth. However, the format of expert data still need to clarify and the correlated code need to be further modified.


# Note about the env for org_airl
if you encounter problem about CPython, try to downgrade it to 0.29.37 and use the command
```
sudo apt-get install python3.7-dev
```
and change the _clamp function at network/policy.py to clamp

it should be all set by then.