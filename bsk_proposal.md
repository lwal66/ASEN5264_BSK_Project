Problem Statement:
    The BSK-RL package has satellite models allowing for simulations of spacecraft systems and orbital mechanics. We will create a simulation and train an optimizer to balance completing our objectives while avoiding significant spacecraft failure events. Our initial formulation of the MDP is as follows:
    
    The full state includes the satellite states, target states and coverage history for all targets. The satellite state includes the position, velocity, and pointing direction for the satellite. The state of each target is the position of the target and its total coverage history. The action set is the discrete list of targets active at any time. The transition matrix is the time/battery cost to move from one city to the next. The reward function is the created by looking at the coverage threshold for each target combined with that targets priority

    Our proposal is to train a PPO (Proximal Policy Optimizer) to learn our spacecraft behavior and balance the 

Levels of Success:
    1- Minimum Working example
            - Prebuilt Objectives
            - Prebuilt Model: Unlimited Power,  No slew time, No Drag

        Success:
            - Completing a single Image Objective

    2- Main Objective
            - Selected Objectives, inc. moving targets
            - Realistic Model: Battery, Slew, Drag, Comm Scheduling

        Success:
            - Completing All Objectives
            - No spacecraft failure

    3- Stretch
            - Multi-Spacecraft Formation
            - Data buffer
            - Error

        Success:
            - Completing All Objectives 
            - maximize collect time
            - No Spacecraft failure