# One-Shot Transfer of Long-Horizon Extrinsic Manipulation Through Contact Retargeting
#### Authors: Albert Wu(amhwu [at] stanford [dot] edu), Ruocheng Wang, Sirui Chen, Clemens Eppner, C. Karen Liu.

This repository contains the code for the paper [One-Shot Transfer of Long-Horizon Extrinsic Manipulation Through Contact Retargeting](https://stanford-tml.github.io/extrinsic-manipulation)

# Installation
## Prerequesites
#### Set up conda virtual environment
This repository is tested in `python 3.8`.
```bash
conda create -n "extrinsic-manip-release-3.8" python=3.8 ipython
```

#### Install `isaacgym`
Obtain the [preview release](https://developer.nvidia.com/isaac-gym) and install following the instructions.

## Installing the main code
```bash
cd contact_demo
pip install -e .
```

## Additional Downloads

### Object scans
Scans of the [*standard* and *short* objects](https://drive.google.com/file/d/1aOAuU38V2uBWVVU6fWbiBFaOQFgIr3Ju/view?usp=drive_link).
We suggest extracting the scans to `assets/urdf/tml/`.

### Pretrained *pushing* reinforcement learning policy
The pretrained weights for the *pushing* reinforcement learning policy is available [here](https://drive.google.com/file/d/1NVBfsILWBO9wfGTKlmU6k6H_7lhEdwkM/view?usp=drive_link)

We suggest placing the pretrained policy in `assets/nn`.


### Precomputed initial-final state pairs for training *pushing* policy
The initial and final state pairs we used to train the *pushing* policy is available [here](https://drive.google.com/file/d/19bCJ-wafBM8971Y3M_4-MIOugbANRrHa/view?usp=drive_link). Alternatively, you can generate the data using the instructions below.

We suggest placing the precomputed state pairs in `assets/state_pairs/`.

# Repository structure
The three main directories relevant to the users are:
`assets, contactdemo, isaacgymenvs`

## `assets`
This directory contains the object scans, pretrained policy, precomputed state pairs, relevant CAD files, and other assets. Refer to the *Additional Downloads* section for what should go in this directory.

## `contactdemo`
This directory contains the non-learning code for the paper. 

`contactdemo/lib/data_generation` contains the code for generating the initial and final states for the pushing task. 

`contactdemo/lib/drake/contact_retargeting` contains the implementation of *Section IV.B Implementing $\sigma$* of the paper using [Drake](https://drake.mit.edu/).

`contactdemo/scripts/demo/record_demo_obj_path.py` is a simple script for recording the demo object path (hardware required).

## `isaacgymenvs`
This is adapted from [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs). It implements the reinforcement learning pipeline for the pushing task and the multi-stage policy.

The class `HardwareVecTask` in `isaacgymenvs/tasks/base/vec_task.py` has been heavily modified to implement Algorithm 1 of the paper. It is the backbone of the multi-stage framework, in addition to still supporting the single-stage training for the pushing policy.

`FrankaYCBPush` is the main "task" for both training the pushing policy.

`FrankaYCBUngraspable` is a slightly modified version of `FrankaYCBPush` for end-to-end training of grasping the ungraspable (Section VI.C of the paper).

# Running the code
**Simulation cannot capture the extrinsic manipulation dynamics. Moreover, lots of our code requires real-time state feedback from the robot and perception system. Thus we only provide instructions for running a very limited subset of our code. Please contact the authors if you want to run the pipeline on hardware.**

## Pushing Data Generation
To generate the initial and end states of objects for training the pushing task, run the following command:

```bash
 PYTHONPATH=. python contactdemo/lib/data_generation/general_pushing_data_generation_scanned_parallel.py --dataset_path $SOME_DIRECTORY --n_examples $N 
```
The dataset will be saved to $SOME_DIRECTORY.pkl, where the data is stored as:
```json
{
  'mustard': [
    {
        'initial_state':  [...], # 7D pose (3D position, 4D quaternion)
        'final_state':  [...] # 7D pose (3D position, 4D quaternion)
    },
    ...
  'xxx': [...]
   ...
}
```
After the data is generated, you can visualize the data by running the following command:
```bash
 PYTHONPATH=. python contactdemo/lib/data_generation/trajectory_visualization_scanned.py --pkl_path $SOME_DIRECTORY.pkl
```

## Visualizing the pretrained `push` policy
After downloading the object scans, pretrained policy, and precomputed state pairs, run
```bash
python train.py task=FrankaYCBPush test=True num_envs=$NUM_ENVS checkpoint=$PATH_TO_PRETRAINED_MODEL wandb_activate=False
```
The number of environments should be set to at least 10 to see all the objects. The $i$th object appears in environments $j$ where $j\%10 = i$.

Please refer to section *Configuration and command line arguments* in `isaacgymenvs_docs/README.md` for details on the command line arguments. Relevant parameters can be found in `isaacgymenvs/cfg/task/FrankaYCBPush.yaml`.