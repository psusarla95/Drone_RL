### Drone_RL
Solving Beamforming problem in radio communications using reinforcement learning between Base station (BS) and an Unmanneed Aerial Vehicle (UAV)

Millimeter waves is one of the key innovations of 5G communications, that can help achieve high data rates using massive Multiple-Input-Multiple-Output (MIMO) radio units and Beamforming technologies. There have only been very traditional approaches in applying radio beamforming which includes complete scanning or scanning at multiple levels. With the increase in complexities to radio communications through 5G technologies, this implementation is an attempt to solve Beamforming problem using Machine Learning especially, Reinforcement Learning. 


## gym_uav

This is a custom environment developed for Drone RL application using OpenAI gym interface. There are two versions of the custom environment inside this directory.


## Source

This directory has library level implementation of MIMO environment, NN_model and Agent implementation of Deep Q Network (DQN) for the Drone-RL problem.

## Test

This directory contains the main function of various tests carried on the developed custom environment and Reinforcement learning solution for Beamforming problem.

## Installation

run `python setup.py` before executing the test code. This helps installing the necessary packages required for this environment.
