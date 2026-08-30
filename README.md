## Concept for the Safety and Feasibility of Receding Horizon Control for Signalized Transportation Systems

In this experiment, I built a reinforcement learning environment for an autonomous vehicle approaching an intersection to evaluate how its acceleration behavior changes depending on whether it knows the signal plan in advance or not.

### Motivation 

As the number of connected and autonomous vehicles (CAVs) in the United States rises, questions about managing passenger/driver/pedestrian/cyclist safety and minimizing traffic and emissions are becoming increasingly relevant. 
Earlier this year, I proposed a concept for game theoretic signal decisions in a traffic network to address these issues. My idea, which used the Jacobian of a transition matrix sending "demand" (self-defined function/subject to change) from the ith intersection to the jth intersection to minimize traffic accumulation, was part of a larger "Receding Horizon" or "Model Predictive" control system. That is, the solution was not deciding a plan to be executed for the next few minutes, rather a computation occurring constantly. 

The stochastic nature of human driving and pedestrian decision-making, I posit, renders predetermined plans reliant on assumptions of human movements not optimal in improving traffic flow. However, unpredictability of signals may pose risks such as drivers caught in dilemma zones, hardware latency making the data for a decision obsolete, unnecessary edge compute costs at RSUs (roadside units) with little demand, etc. In this work, I primarily address compatibility between predictable/unpredictable intersections and vehicles.

## Design

### Data

I used data from the public Utah Department of Transportation dataset, containing Signal Plan and Timing (SPaT) messages from several intersections and their signal groups, accessed with the Socrata API. The dataset is not included in the repository due to its large size. My data folder contains the raw python code I used to clean messy JSON messages and columns from this dataset to useable data for signal timing in a reinforcement learning environment. 

### Environment

The RL environment provided the agent with an action space of a scalar between -1 and 1, its jerk control. -1 is taken to mean the agent wants to press the brakes as hard as possible, while 1 means it wants to accelerate as fast as possible. This action is normalized to a feasible jerk governed by min/max acceleration and velocities. The chosen jerk then adjusts the state (pos, vel, acc) of the vehicle relative to the signal via matrix multiplication ($x_{t+1} = A x_t + B u_t$). 

The agent sees an observation space of the (pos, vel, acc) vector, the current signal, and the future signal n timesteps from now (which is what we are testing). 

The agent receives reward for successfully crossing the intersection, loses reward when committing a red light violation, and is penalized with the square of acceleration and the square of jerk multiplied by tunable parameters, to encourage the agent to ensure comfortable driving. 

## Preliminary Diagnostics

### Model

The training diagnostics have shown stable, sensible results for the agent's training. The reward starts in the high negatives and stabilizes close to zero, entropy loss decreased toward a stable, nonzero value, the explained variance stabilized over .9. 

The Kullback-Leibler divergence is generally bounded but has relatively large fluctuations, which I initially suspect is due to tail cases of signals lasting much longer or much shorter than normal due to possibly high variability in the signals or errors in the dataset. I aim to additionally evaluate whether the model can become more confident in its actions and better learn to navigate tail case situations if also trained on high variance-injected synthetic data. 

### Movements

Thus far, I have evaluated the model's movements by how many red light violations and hard braking events (< -13.0 ft/s^2) it commits during training. I currently plan on adjusting the reward function due to large numbers of hard braking events, but I plan on still avoiding explicitly penalizing it to prevent the model from memorizing that value in training. 

## Initial Results

I have tested the model while trained on 0, 1, 3, 5, 8 seconds of future data of the signal, and there is not sufficient evidence to conclude the agent performs better on larger lookahead horizons: $p = 0.5452906 > \alpha = 0.05$; fail to reject the null hypothesis. Nonetheless, this initial result should be heavily scrutinized. Later actions should include among others modifying and optimizing the reward function and increasing the sample size of the reinforcement learning trainings (my poor mac has requested I do this on another computer lol). 

## Other Project Details

I am testing RHC systems with model size hardware in the Cornell Maker Lab. We will export trained policy graphs and model weights to C++ with ONNX to evaluate the systems at the edge. This should introduce other considerations including latency, noisy sensors, and black box decisions to address as well. 



