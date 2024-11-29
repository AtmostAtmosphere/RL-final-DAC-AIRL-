# RL-final-DAC-AIRL-
Use python=3.7 to download the specified version of modules in requirements.txt
(If not, serveral bugs may appear: new version has no "env(seed)", "env.step" return 5 instead of 4 values, etc)
Note:
1. If your desktop has no CUDA, change all the torch.device("cuda" if args.cuda else "cpu") to torch.device("cpu")
2. 
