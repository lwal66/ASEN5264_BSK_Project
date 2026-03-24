Problem Statement:
    The BSK-RL package has satellite models allowing for simulations of spacecraft systems and orbital mechanics. We will create a simulation and train an optimizer to balance completing our objectives while avoiding significant spacecraft failure events. Our initial formulation of the MDP is as follows:
    
    The full state includes the satellite states, target states and coverage history for all targets. The satellite state includes the position, velocity, and pointing direction for the satellite. The state of each target is the position of the target and its total coverage history. The action set is the discrete list of targets active at any time. The transition matrix is the time/battery cost to move from one city to the next. The reward function is the created by looking at the coverage threshold for each target combined with that targets priority

    Our proposal is to train a PPO (Proximal Policy Optimizer) to learn our spacecraft behavior and balance the cost of moving between objectives vs time and battery cost.

Levels of Success:
    1- Minimum Working example
            We will use the prebuild BSK objectives and models for simulating our satellite, and simplify
            several parameters such as unlimited power, simplified constant slew rate, and no drag. This is to allow us to achieve a working model and minimally train our PPO to optimize the problem. In this stage we define success by successfully utilizing the prebuilt reward function for the model.

    2- Main Objective
            For our main objective, we will introduce constraints to our model such as prioritized objectives, limited battery, slew, drag. This is to increase fidelity in our system, in order to simulate a more accurate and complex model for our optimizer to learn. For our main objective we define success as completing all prioritized objectives while avoiding a spacecraft failure event. Completing an objective is defined as meeting a total coverage collect time threshold as required.

    3- Stretch Goals
            For our stretch goals, we define an additional series of possible constraints and parameters to further explore the complexity of our model. The targeted constraints we will prioritize are as follows:
                - Moving Targets
                - Communication scheduling
                - Data buffer
                - Resilience for inconsistent target location
                - Multi-Spacecraft Formation
            For our stretch goals, we define success as completing all prioritized objects, as well as maximizing median collect time for each objective, while avoiding a spacecraft failure event.