#!/usr/bin/env python3

"""
Reference paper: J. Bodempudi, B. S. Sairam, M. Haritha, S. R. Mattu and A. Chockalingam, 
        "A Reinforcement Learning Framework for Resource Allocation in Uplink Carrier Aggregation in the Presence of Self Interference,"
        in IEEE Transactions on Machine Learning in Communications and Networking, vol. 3, pp. 1265-1286, 2025,
        doi: 10.1109/TMLCN.2025.363324

Date: 2025-11-26

Description:
This code implements the RL algorithm/methodology described in our paper mentioned above.
It serves as a starting point for understanding and building upon our approach.

Note:
- This implementation demonstrates the RL framework for resource allocation in uplink carrier aggregation.
- Additional code modification and parameter optimization are required to generate the results and figures shown in the paper.
"""



# TensorFlow Agents imports
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ddpg import (
    ddpg_agent,
    critic_network,
    actor_network,
    critic_rnn_network,
    actor_rnn_network,
)
from tf_agents.specs import tensor_spec, array_spec
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents import utils
from tf_agents.trajectories import time_step as ts

# Random number generation
from random import randint, randrange, sample
from random import random as rnd

# Command-line interface tools
from fire import Fire
from argparse import Namespace

# Standard library imports
import sys
import math
import time

# Data visualization and numerical computing
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy

options = Namespace(
    pmax_dB=-3,  # Maximum transmit power level for UEs
    G_dB = 30,  # Gain of PA 
    dQoS=0.15,  # Delay requirement
    M=2,         # No of CCs    
    BW_PCC=5*1e6,   # Bandwidth of PCC
    BW_SCC=18e4, # Bandwidth of SCC
    K=2,       # No. of users per gNB
    B=2,        # No. of gNBs
    data_per_T=1e3,
    state=np.zeros(()), # State of UEs
    random=False, # Flag to indicate if random 
)

K=2 # No of UEs per gNB
NR=2 # No. of gNBs

# UE locations - gNB1 has UEs at 10m and 20m, gNB2 has UEs at 15m and 30m
UE_locations=np.array([10, 20, 15, 30])  # First K for gNB1, next K for gNB2

# Interference locations - for gNB1 UEs, interferers are at 70m and 85m
# For gNB2 UEs, interferers are at 80m and 90m
UE_interference_locations=np.array([70, 85, 80, 90])

M=2  # No of CCs 
d_max = 50 # Cell radius (in m)
MRB=50 # Total No of RBs in a CC
alpha_2 = 0.1417 # Non-linearity model co-efficient
L_coupling_dB = 35 # Self_coupling loss
L_coupling = math.pow(10,0.1*(L_coupling_dB))  # Self_coupling loss in linear
noise_var=1e-10  # Noise variance
action_space=np.zeros((pow(2,(M-1)*K),(M-1)*K)) # Action space

# Index of CC causing SI in UE  # 1 indicates SCC1 (CC2), 2 indicates SCC2 (CC3) and so on
MSD_CC_idx = np.array([[1],[1]]) # SI index of CCs
MSD_causing_CC = 1 # CC2 causes SI

########################## Set the flag as per scenario ###########################
SA_finer = 1# Set this flag for "SI,SA,Res25"
SA =0# Set this flag for "SI,SA,Res50"
HA = 0 # Set this flag for "HA"

if HA == 1:
    n_m = 1
    c = 0
else:
    if SA_finer == 1:
        n_m = 2 # Set n_m = 2 for "SI,SA,Res25"
        c = 1
    elif SA == 1:
        n_m = 1
        c = 1
    else:   # No SI scenario
        n_m = 1
        c = 0 

####################################################################################


active_UE_indices = list(range(K))  # active UEs in the network

action_space_extended = np.zeros((pow(2,((M-1)+(n_m-1))*K),((M-1)+(n_m-1))*K)) # Extended action space 
for i in range(pow(2,((M-1)+(n_m-1))*K)):
    s_extended = bin(i)[2:]
    action_space_extended[i,:]=list(map(int,list('0' * (((M-1)+(n_m-1))*K - len(s_extended)) + s_extended)))

# In HA, shutdown CC causing SI
if HA == 1:
    for i in range(K):
        action_space_extended[:,i] = np.zeros(len(action_space_extended))

# Gain calculation
def gain(G_dB):
    return (math.pow(10,0.1*(options.G_dB)))          
    
# Signal interference noise ratio          
def get_SINR(UE, power, noise_var, path_loss, path_loss_interferer, CC, RB, G_dB):
    numerator = gain(G_dB) * power[UE * M * MRB + CC * MRB + RB] * path_loss
    denominator = (gain(G_dB) * power[UE * M * MRB + CC * MRB + RB] * path_loss_interferer +
                   noise_var)
    return numerator / denominator

# Path loss calculation for gNB1
def get_pathloss(UE, gNB):
    # UE locations for gNB1 are first K elements
    distance = UE_locations[UE] ** 2
    return (1/math.pow((4*math.pi*distance)/0.0857143,2))

# Path loss calculation for gNB2
def get_pathloss2(UE, gNB):
    # Interference locations for gNB1 UEs are first K elements in interference array
    if gNB == 0:  # gNB1's UEs
        distance = UE_interference_locations[UE] ** 2
    else:  # gNB2's UEs
        distance = UE_interference_locations[K + UE] ** 2
    return (1/math.pow((4*math.pi*distance)/0.0857143,2))

# Rate calculation
def get_rate_UE(UE, RB, power, G_dB):
    sum_rate = 0
    for j in range(M):  # Iterate over CCs
        for k in range(MRB):  # Iterate over RBs
            if RB[UE * M * MRB + j * MRB + k] == 1:
                path_loss = get_pathloss(UE, UE//K)  # Path loss for serving gNB
                path_loss_interferer = get_pathloss2(UE, (UE//K + 1) % NR)  # Interference from other gNB
                SINR = get_SINR(UE, power, noise_var, path_loss, path_loss_interferer, j, k, G_dB)
                sum_rate += options.BW_SCC * math.log2(1 + SINR)
    return sum_rate

# Delay calculation
def get_delay_UE(UE, data, RB, power,G_dB):
    return (data/get_rate_UE(UE,RB,power,G_dB)) 

# Round Robin algorithm for Resource Block(RB) allocation in CCs
def RRalgorithm(cont_action,discrete_action):
    RB=np.zeros([M*MRB*K]) 
    power=np.zeros([M*MRB*K])
    x=0
    
   
    #### for PCC ####
    j=0
    for i in range(K):
        k = np.array(range(MRB//K))
        RB[i*MRB*M+j*MRB+k]=1    
    

    # Needs to generalise the case for K UEs per gNB

    #### for SCCs ####
    for CC in range(1,M): # Since PCC is 0th CC; SCC starts with CC = 1 to M-1
        if MSD_causing_CC == CC:
            for i in range(K):
                for xp_idx in range(n_m):
                    if discrete_action[i*((M-1)+(n_m-1))+xp_idx+(CC-1)]==1:
                        if MRB%(K*n_m) == 0:
                            xp_len = int(math.ceil(MRB/(K*n_m)))                            
                        else:
                            xp_len = int(abs((MRB/n_m)*(xp_idx+(CC-1)) - math.ceil(MRB/(K*n_m)))) 
                        kd = np.array(range(xp_len))
                        RB[i*MRB*M+CC*MRB+xp_idx*(MRB//n_m)+kd]=1

        else: # RBs allocation in SCC not causing SI
            if MSD_causing_CC > CC:
                CC_idx = CC
            else:
                CC_idx = CC + (n_m-1)
            
            x=0
            
            for i in range(CC_idx-1,((M-1)+(n_m-1))*K,(M-1)+(n_m-1)): 
                x=x+discrete_action[i]  
            for i in range(K):
                if discrete_action[i*((M-1)+(n_m-1))+(CC_idx-1)]==1:
                    y=int(MRB//x) 
                    v=0
                    for k in range(0,MRB):  
                        if RB[i*MRB*M+CC*MRB+k]==0.:
                            v=k
                            break
                    kd = np.array(range(y))
                    RB[i*MRB*M+CC*MRB+kd+v]=1
    for i in range(K):
        sum1=sum(RB[i*MRB*M:(i+1)*MRB*M-1])
        for k in range(M*MRB):
            if RB[i*MRB*M+k]==1:    
                power[i*MRB*M+k]=cont_action[0,i]/sum1   
    return (RB,power)


# Computing Self Interference Power (P_SI)
def self_interference_power(RB,power): 
    #initializing with zeros
    A_sq_in = np.zeros([K*M]) # (A_i,j)^2
    P_SI = np.zeros([K*M])    # (p_2H)_Tx,out : Power of 2nd harmonic at Tx. output  
    P_SI_Rx = np.zeros([K])   # (p_SI)_Rx,in : SI power at the input of Rx
    for i in range(K):
        for j in MSD_CC_idx[i]: # Selected CC index => CC(j+1)
            A_sq_in[i*M+j] = (2*sum(power[i*MRB*M+j*MRB:i*MRB*M+(j+1)*MRB-1]));
            P_SI[i*M+j] = pow(alpha_2,2)*pow(A_sq_in[i*M+j],2)/8;
        
        P_SI_Rx[i] = P_SI[i*M+MSD_CC_idx[i][0]]/L_coupling

    return P_SI_Rx

# Reward calculation 
def get_sumrate(cont_action,discrete_action):
    [RB,power] = RRalgorithm(cont_action,discrete_action)
    P_SI_Rx = self_interference_power(RB,power)

    # Total throughout for all UEs in the network
    R_a = sum(
                [
                    get_rate_UE(UE,RB,power,options.G_dB)
                    for UE in range(options.K)
                ]
            )
    #### Proposed penalty function computation ####

    # theta^i values corresponds to (p_SI)^i value above and sensitivity degradation (SD)
    th_1_UE1 = 1e-13   # SD is 3dB, corresponding
    th_2_UE1 = 3.162*1e-13 # SD is 6dB
    R_s_UE1 = 0.625*1e7    # Indicates \Omega^i in the penalty function


    P_SI_UE1_CC2 = P_SI_Rx[0]
    if P_SI_UE1_CC2 <= th_1_UE1:
        R_p1 = 0
    elif P_SI_UE1_CC2 > th_1_UE1 and P_SI_UE1_CC2 < th_2_UE1:
        R_p1 = R_s_UE1*(P_SI_UE1_CC2 - th_1_UE1)/(th_2_UE1 - th_1_UE1)
    else:
        R_p1 = R_s_UE1

    d_1_norm = pow((UE_locations[0]/d_max),2) # Normalised distance
    R_p = (R_p1/d_1_norm)

    if K == 2: #Need to finetune the values for more than 2 UE
        th_1_UE2 = th_1_UE1
        th_2_UE2 = th_2_UE1

        R_s_UE2 = R_s_UE1   # Fine tune this parameter for multi-UE

        P_SI_UE2_CC2 = P_SI_Rx[1]

        if P_SI_UE2_CC2 <= th_1_UE2:
            R_pi = 0
        elif P_SI_UE2_CC2 > th_1_UE2 and P_SI_UE2_CC2 < th_2_UE2:
            R_pi = R_s_UE2*(P_SI_UE2_CC2 - th_1_UE2)/(th_2_UE2 - th_1_UE2)
        else:
            R_pi = R_s_UE2

        d_2_norm = pow((UE_locations[1]/d_max),2)
        R_p = (R_p1/d_1_norm) + (R_pi/d_2_norm)

    return  (R_a-c*R_p)     # Reward function

# Resetting state vector
def reset_state():
    options.state = tf.constant(np.zeros((K,2)))

# CC selection algorithm
def get_best_discrete_action(cont_action):
    highest_reward = 0
    best_action = np.zeros((1,((M-1)+(n_m-1))*K))

    # Out of all valid actions, choose best 
    for action in range(pow(2,((M-1)+(n_m-1))*K)):
        chosen_action = action_space_extended[action,:]
        reward = get_sumrate(cont_action,chosen_action);
        if reward >= highest_reward:
            best_action = chosen_action
            highest_reward = reward

    return (tf.reshape(best_action,(1,((M-1)+(n_m-1))*K)))

# State vector is updated for every time step based on whether UE meets QoS delay requirement
def get_state(dQoS,cont_action,discrete_action):
    [RB,power]=RRalgorithm(cont_action,discrete_action)
    state=np.zeros((K,))
    for i in range(K):
        if dQoS<get_delay_UE(i, options.data_per_T, RB,power,options.G_dB):
            state[i]=0
        else:
            state[i]=1  
    return state

#################################### MAIN #####################################################
def main(
    pmax_dB=options.pmax_dB,
    G_dB = options.G_dB,
    dQoS=options.dQoS,
    M=options.M,
    BW_PCC=options.BW_PCC,
    BW_SCC=options.BW_SCC,
    K=options.K,
    B=options.B,
    state=options.state,
    random=options.random,
    data=options.data_per_T,):

    options.pmax_dB = pmax_dB
    options.G_dB =G_dB
    options.dQoS = dQoS
    options.M = M
    options.BW_PCC= BW_PCC
    options.BW_SCC = BW_SCC
    options.K = K
    options.B = B
    options.random = random
    options.state=state
    options.data_per_T=data

    class DurpEnv(py_environment.PyEnvironment):
        def __init__(self, gNB):
            # Initialize the environment with a specific gNB
            self._gNB = gNB

            self._action_spec = array_spec.BoundedArraySpec(
                shape=(options.K*(M+(n_m-1)),),
                dtype=np.float64,
                minimum=0,
                maximum=math.pow(10,0.1*(options.pmax_dB-options.G_dB)), 
                name="action",
            )

            # specifies the state input to network
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(K,),
                minimum=0,
                maximum=1,
                dtype=np.float64,
            )

        def action_spec(self):
            return self._action_spec

        def observation_spec(self):
            return self._observation_spec
        
        def _reset(self):
            # Reset the environment
            # Return the initial state (0, 0) as a restart time step
            return ts.restart(np.reshape(np.zeros(K),(K,)))
        
        def _step(self, action):
            cont_action=tf.gather(action,indices=range(K),axis=0)
            chosen_action=tf.gather(action,indices=range(K,K*(M+(n_m-1))),axis=0)

            cont_action = tf.reshape(cont_action,[1,-1])

            reward=get_sumrate(cont_action,chosen_action)  
            state=get_state(dQoS,cont_action,chosen_action)  
            return ts.transition(state, reward=reward)
        
    # Set the learning rate for the optimization algorithm
    learning_rate = 0.01
    # Create an Adam optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Initialize empty lists to store multiple environments and agents
    environments = []
    agents = []

    for gNB in range(options.B):
        # Create a DurpEnv environment for the current gNB
        durp_env = DurpEnv(gNB)   
        # Wrap the Python environment in a TensorFlow environment
        train_env = tf_py_environment.TFPyEnvironment(durp_env) # environment for a gNB
        
        actor_net = actor_network.ActorNetwork(
        train_env.observation_spec(), 
        train_env.action_spec()
        )

        critic_net = critic_network.CriticNetwork(
        (train_env.observation_spec(), train_env.action_spec()), 
        )   
        
        agent = ddpg_agent.DdpgAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        critic_network=critic_net,
        actor_network=actor_net,
        critic_optimizer=optimizer,
        actor_optimizer=optimizer,
        train_step_counter = tf.Variable(0))
        
        # Initialize the agent
        agent.initialize()
        # Add the agent and environment to their respective lists
        agents.append(agent)  
        environments.append(train_env) 

    
    num_iterations =100 # Total number of training iterations
    intermediate_iterations = 1  # Number of iterations between evaluations 
    collect_steps_per_iteration = 100 # Steps to collect data per training iteration
    batch_size = 500 # # Batch size for training
   

    power=np.zeros([num_iterations,collect_steps_per_iteration,M*MRB*K,NR])
    RB=np.zeros([num_iterations,collect_steps_per_iteration,M*MRB*K,NR])
    actions=np.zeros([num_iterations,collect_steps_per_iteration,K*(M+(n_m-1)),NR])
    rewards=np.zeros([num_iterations,collect_steps_per_iteration,NR])


    def collect_step(state,environment, policy, agent_policy,buffer, gNB, iteration,step):
        # Get the current time step from the environment
        time_step = environment.current_time_step()
        print("step is",step)
        P=agent_policy.action(time_step).action # The agent's policy suggests 
        # a continuous action based on the current state.
        
        best_discrete_action = get_best_discrete_action(P)
        
        # Extracts the first K components of the continuous action 
        #  corresponding to power allocation. 
        best_continous_action=tf.gather(agent_policy.action(time_step).action,indices=range(K),axis=1)   
        # Use the policy to choose an action based on the current time step
        action_step = policy.action(time_step)

        # Espilon greedy algorithm 
        if np.random.rand(1) < 0.9*(1-(iteration /num_iterations)):
            action_step = action_step.replace(action=tf.concat([best_continous_action,[action_space_extended[np.random.randint(0,len(action_space_extended)-1),:]]],1))                                                               

        else:
            action_step = action_step.replace(action=tf.concat([best_continous_action,best_discrete_action],1))
        # Take a step in the environment using the chosen action
        next_time_step = environment.step(action_step.action)
        # Update the global state with the new observation.
        options.state = next_time_step.observation
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        # Store the experience in the replay buffer for future learning.
        buffer.add_batch(traj)
        return (traj.action,traj.reward)
    
    def collect_data(state,env, policy,agent_policy, buffer, steps, gNB,iteration):
        actions1=np.zeros([collect_steps_per_iteration,K*(M+(n_m-1))])
        rewards1=np.zeros([collect_steps_per_iteration])

        #This loop calls collect_step for each step and 
        # stores the returned values in the corresponding arrays.
        for step in range(0, steps):
            actions1[step,:],rewards1[step]=collect_step(state,env, policy, agent_policy,buffer, gNB,iteration,step)
        return (actions1,rewards1)
    

    replay_buffers = [
    tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agents[i].collect_data_spec, batch_size=1
    ) for i in range(NR)
                     ]

    reset_state()
    
    # This creates a TensorFlow dataset from the replay buffer for efficient training.
    # This creates an iterator for the dataset.
    iterators = [
    iter(replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)) for replay_buffer in replay_buffers
                ]
    # This initializes an array to store returns for each evaluation.
    

    for i in range(num_iterations):
        # Resets the environment state at the beginning of each iteration.
        reset_state()
        print("episode",i)

        
        for j in range(intermediate_iterations):
            for k in range(len(agents)):
                agent=agents[k]
                env=environments[k]
                actions[i,:,:,k],rewards[i,:,k]=collect_data(state,env, agent.collect_policy, agent.policy,replay_buffers[k], collect_steps_per_iteration, k, i)
               
                
        for k in range(len(agents)):
            agent = agents[k]
            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterators[k])
            train_loss = agent.train(experience).loss
            print("Loss:", train_loss)

    print("for debug")
    # Saves actions, rewards, UE locations into .mat file
    scipy.io.savemat('two_cell_output_RR.mat',mdict={'actions':actions,'rewards':rewards,'UE_locations':UE_locations,'RB':RB,'power':power})
if __name__ == "__main__":
    Fire(main)