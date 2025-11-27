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

from random import randint, randrange, sample
from random import random as rnd
from fire import Fire
from argparse import Namespace
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy



K=2 # No of UE's
UE_locations=np.array([25,35]) # User locations in the Network (m)
d_max = 50 # Cell radius (in m)
M=2   # No of CCs
MRB=50 # Total no. of RBs in a CC (N_max)
c_2 = 0.1417  # Non-linearity model co-efficient
L_coupling_dB = 35 # Self-coupling loss
L_coupling = math.pow(10,0.1*(L_coupling_dB))  
sigma2_dBm = -70
noise_var = 1e-3*math.pow(10,0.1*(sigma2_dBm))  # Noise variance

# Index of CC causing SI in UE  # 1 indicates SCC1 (CC2), 2 indicates SCC2 (CC3) and so on
MSD_CC_idx = np.array([[1],[1]]) # shape: (K,1) 
MSD_causing_CC = 1 #CC2 causes SI

options = Namespace(
    pmax_dB=-3,  
    G_dB = 30, # Gain of PA
    dQoS=0.15,  # Delay requirement
    M=M,        # No. of CCs
    BW_PCC=5*1e6, # BW of PCC (CC1)
    BW_SCC=18e4,  # BW of SCC 
    K=K,      # No. of UEs in the network
    B=1,        # no of gNBs
    data_per_T=1e3,
    state=np.zeros((K,1)), # states of UE
    random=False,
)

########################## Set the flag as per scenario ###########################
SA_finer = 0 # Set this flag for "SI,SA,Res25", "SI,SA,Res10" and so on
SA = 1 # Set this flag for "SI,SA,Res50"
HA = 0 # Set this flag for "HA"
# If none of the above flags are set, then "No SI"

if HA == 1:
    n_m = 1
    c = 0
else:
    if SA_finer == 1:
        n_m = 2     # Set n_m = MRB/V for "SI,SA,ResV"
        c = 1
    elif SA == 1:
        n_m = 1
        c = 1
    else:   # No SI scenario
        n_m = 1
        c = 0 

####################################################################################

active_UE_indices = list(range(K))  # active UEs in the network
episode_count = 0  # UE exits and re-enters network depending at particular values


# Action space 
action_space_extended = np.zeros((pow(2,((options.M-1)+(n_m-1))*K),((options.M-1)+(n_m-1))*K))
for i in range(pow(2,((options.M-1)+(n_m-1))*K)):
    s_extended = bin(i)[2:]
    action_space_extended[i,:]=list(map(int,list('0' * (((options.M-1)+(n_m-1))*K - len(s_extended)) + s_extended)))

# In HA, shutdown CC causing SI
if HA == 1:
    for i in range(K):
        action_space_extended[:,i] = np.zeros(len(action_space_extended))

# Gain computation
def gain(G_dB):
    return (math.pow(10,0.1*(G_dB)))

# SINR computation
def get_SINR(UE,power,noise_var,path_loss,CC,RB,G_dB):
    return (gain(G_dB)*power[UE*options.M*MRB+CC*MRB+RB]*path_loss)/(noise_var)

# Pathloss computation 
def get_pathloss(UE):
    return 1/math.pow((4*math.pi*UE_locations[UE])/0.0857143,2)

# UE Rate computation
def get_rate_UE(UE,RB,power,G_dB):
    sum=0
    for j in range(options.M):
        for k in range(MRB):
            if RB[UE*options.M*MRB+j*MRB+k]==1:
                sum=sum+options.BW_SCC*math.log2(1+get_SINR(UE,power,noise_var,get_pathloss(UE),j,k,G_dB))
    return sum

# UE delay computation
def get_delay_UE(UE, data, RB, power,G_dB):
    return (data/get_rate_UE(UE,RB,power,G_dB)) 

# RB allocation algorithm
def RRalgorithm(cont_action,discrete_action):
    RB=np.zeros([options.M*MRB*K]) 
    power=np.zeros([options.M*MRB*K])    
    K_act = len(active_UE_indices)

    #### for PCC ####
    j=0
    for i in active_UE_indices:
        k = np.array(range(MRB//K_act))
        RB[i*MRB*options.M+j*MRB+k]=1    
        
    #### for SCCs ####
    for CC in range(1,options.M): 

        if SA_finer == 1 and MSD_causing_CC == CC: # finer RB allocation in SCC causing SI during 'Res25', 'Res10' cases

            for idx in range(n_m):
                UE_CC_nm = np.zeros(K)
                for i in active_UE_indices:

                    UE_CC_nm[i] = discrete_action[i*((M-1)+(n_m-1))+idx+(CC-1)]

                if MRB%(K_act*n_m) == 0:
                    n_a = int(MRB/(K_act*n_m))  # per CC nm length = 5

                else:
                    n_a = int(math.ceil(MRB/(K_act*n_m)))


                # Needs to generalise the case for K UEs
                if n_m == 5:

                    if UE_CC_nm[0] == 1 and UE_CC_nm[1] == 1:
                        kd = np.array(range(n_a))

                        for i in active_UE_indices:
                            RB[i*MRB*M+CC*MRB+idx*(MRB//n_m)+kd]=1

                    if UE_CC_nm[0] == 1 and UE_CC_nm[1] == 0:
                        kd = np.array(range(2*n_a))

                        RB[0*MRB*M+CC*MRB+idx*(MRB//n_m)+kd]=1                           
                    
                    if UE_CC_nm[0] == 0 and UE_CC_nm[1] == 1:
                        kd = np.array(range(2*n_a))

                        RB[1*MRB*M+CC*MRB+idx*(MRB//n_m)+kd]=1  

                    if UE_CC_nm[0] == 0 and UE_CC_nm[1] == 0:
                        kd = np.array(range(2*n_a))

                if n_m == 2:

                    if UE_CC_nm[0] == 1 and UE_CC_nm[1] == 1:
                        kd = np.array(range(n_a))
                        RB[0*MRB*M+CC*MRB+idx*(MRB//n_m)+kd]=1

                        kd = np.array(range(n_a-1))
                        RB[1*MRB*M+CC*MRB+idx*(MRB//n_m)+kd]=1

                    if UE_CC_nm[0] == 1 and UE_CC_nm[1] == 0:
                        kd = np.array(range((MRB//n_m)))

                        RB[0*MRB*M+CC*MRB+idx*(MRB//n_m)+kd]=1          

                    if UE_CC_nm[0] == 0 and UE_CC_nm[1] == 1:
                        kd = np.array(range((MRB//n_m)))

                        RB[1*MRB*M+CC*MRB+idx*(MRB//n_m)+kd]=1    

                    if UE_CC_nm[0] == 0 and UE_CC_nm[1] == 0:
                        kd = np.array(range(2*n_a))

        else: # RB allocation to SCCs in 'No SI', 'HA', 'Res50' cases

            x=0
            x = sum(discrete_action[i*((M-1)+(n_m-1))+(CC-1)] for i in active_UE_indices)
            for i in active_UE_indices:
                if discrete_action[i*((M-1)+(n_m-1))+(CC-1)]==1:
                    y=int(MRB//x) 
                    v=0
                    for k in range(0,MRB):  
                        if RB[i*MRB*M+CC*MRB+k]==0.:
                            v=k
                            break
                    kd = np.array(range(y))
                    RB[i*MRB*M+CC*MRB+kd+v]=1

    for i in active_UE_indices:
        sum1=sum(RB[i*MRB*options.M:(i+1)*MRB*options.M-1])
        for k in range(options.M*MRB):
            if RB[i*MRB*options.M+k]==1:   
                power[i*MRB*options.M+k]=cont_action[0,i]/sum1
    return (RB,power)

# Self interference power computation
def self_interference_power(power): 

    A_sq_in = np.zeros([K*options.M]) # (A_i,j)^2
    P_SI = np.zeros([K*options.M])   # (p_2H)_Tx,out : Power of 2nd harmonic at Tx. output
    P_SI_Rx = np.zeros([K]) # (p_SI)_Rx,in : SI power at the input of Rx

    for i in active_UE_indices:
        for j in MSD_CC_idx[i]: # compute p_SI for CCs causing SI  
            A_sq_in[i*options.M+j] = (2*sum(power[i*MRB*options.M+j*MRB:i*MRB*options.M+(j+1)*MRB-1]));
            P_SI[i*options.M+j] = pow(c_2,2)*pow(A_sq_in[i*options.M+j],2)/8;            
        
        P_SI_Rx[i] = P_SI[i*options.M+MSD_CC_idx[i][0]]/L_coupling

    return P_SI_Rx

# Reward function computation
def get_sumrate(cont_action,discrete_action):
    [RB,power] = RRalgorithm(cont_action,discrete_action)
    P_SI_Rx = self_interference_power(power)

    # Total throughout for all UEs in the network
    R_a = sum(
                [
                    get_rate_UE(UE,RB,power,options.G_dB)
                    for UE in active_UE_indices
                ]
            )
    
    #### Proposed penalty function computation ####

    # theta^i values corresponds to (p_SI)^i value above and sensitivity degradation (SD)
    th_1_UE1 = 1e-13 # SD is 3dB, corresponding
    th_2_UE1 = 3.162*1e-13 # SD is 6dB

    #### Needs to be tuned such that it penalizes high SI power levels
    R_s_UE1 = 0.5*1e7 # Indicates \Omega^i parameter in the penalty function


    P_SI_UE1_CC2 = P_SI_Rx[0]
    if P_SI_UE1_CC2 <= th_1_UE1:
        R_p1 = 0
    elif P_SI_UE1_CC2 > th_1_UE1 and P_SI_UE1_CC2 < th_2_UE1:
        R_p1 = R_s_UE1*(P_SI_UE1_CC2 - th_1_UE1)/(th_2_UE1 - th_1_UE1)
    else:
        R_p1 = R_s_UE1

    d_1_norm = pow((UE_locations[0]/d_max),2)
    R_p = (R_p1/d_1_norm)

    if K >= 2: 

        th_1_UE_i = th_1_UE1
        th_2_UE_i = th_2_UE1

        R_s_UE_i = 1e7 # Fine tune this parameter for multi-UE such that it penalizes high SI power levels

        Multi_UE_penalty = 0

        for UE_idx in range(1,K):
            P_SI_UE2_CC2 = P_SI_Rx[UE_idx]
            d_i_norm = pow((UE_locations[UE_idx]/d_max),2)

            if P_SI_UE2_CC2 <= th_1_UE_i:
                R_pi = 0
            elif P_SI_UE2_CC2 > th_1_UE_i and P_SI_UE2_CC2 < th_2_UE_i:
                R_pi = R_s_UE_i*(P_SI_UE2_CC2 - th_1_UE_i)/(th_2_UE_i - th_1_UE_i)
            else:
                R_pi = R_s_UE_i

            Multi_UE_penalty += (R_pi/d_i_norm)

        R_p = (R_p1/d_1_norm) + Multi_UE_penalty

    return  (R_a-c*R_p) # Reward function

# Resetting state vector
def reset_state():

    # ####### Online adaptation when UE exits and re-enters network #########################
    # global active_UE_indices
    # Specify when UE exits and re-enters network and update the current number of active UEs in the network
    # #######################################################################################

    K_active = len(active_UE_indices)
    options.state = tf.constant(np.zeros((K_active,1))) 

# CC selection algorithm
def get_best_discrete_action(cont_action):

    highest_reward = 0
    best_action = np.zeros((1,((options.M-1)+(n_m-1))*K))

    # Create a mask for valid actions based on active UEs.
    global valid_actions_mask
    valid_actions_mask = np.copy(action_space_extended)

    # Set the corresponding indices to zero for removed UEs.
    for removed_ue in range(K):
        if removed_ue not in active_UE_indices:
            start_idx = removed_ue*(options.M-1)
            end_idx = ((removed_ue+1)*(options.M-1))
            valid_actions_mask[:, range(start_idx,end_idx)] = 0.0

    # Out of all valid actions, choose best action
    for action in range(valid_actions_mask.shape[0]):
        chosen_action = valid_actions_mask[action,:]
        reward = get_sumrate(cont_action,chosen_action)
        if reward >= highest_reward:
            best_action = chosen_action
            highest_reward = reward

    return (tf.reshape(best_action,(1,((options.M-1)+(n_m-1))*K)))

# State vector is updated for every time step based on whether UE meets QoS delay requirement 
def get_state(dQoS,cont_action,discrete_action):
    [RB,power]=RRalgorithm(cont_action,discrete_action)
    state=np.zeros((K,))
    for i in active_UE_indices:
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
    options.G_dB = G_dB    
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
        def __init__(self, gNB): # Initialize the environment with a specific gNB
            self._gNB = gNB
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(options.K*(options.M+(n_m-1)),),
                dtype=np.float64,
                minimum=0,
                maximum=math.pow(10,0.1*(options.pmax_dB-options.G_dB)),  
                name="action",
            )

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
            return ts.restart(np.reshape(np.zeros(K),(K,)))
        
        def _step(self, action):
            cont_action=tf.gather(action,indices=range(K),axis=0).numpy()
            chosen_action=tf.gather(action,indices=range(K,K*(options.M+(n_m-1))),axis=0).numpy()

            if len(active_UE_indices) < K:
                for UE_idx in range(K):
                    if UE_idx not in active_UE_indices:
                        cont_action[UE_idx] = 0
                        chosen_action[UE_idx*(options.M-1+n_m-1):(UE_idx+1)*(options.M-1+n_m-1)-1] = 0

            cont_action = tf.convert_to_tensor(cont_action)
            chosen_action = tf.convert_to_tensor(chosen_action)

            cont_action = tf.reshape(cont_action,[1,-1])

            reward=get_sumrate(cont_action,chosen_action)  
            state=get_state(dQoS,cont_action,chosen_action)  
            for i in range(K):
                if i not in active_UE_indices:
                    state[i] = 0

            return ts.transition(state, reward=reward)

    learning_rate = 0.01 # Learning rate 0.01
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Adam optimizer
    environments = []
    agents = []

    for gNB in range(options.B):
        durp_env = DurpEnv(gNB)   
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
   
        agent.initialize()
        agents.append(agent)  # agents intiliazation
        environments.append(train_env) # state initialization

    num_iterations = 100 # no. of training episodes
    intermediate_iterations = 1 # no. of iterations between evaluations 
    eval_interval = 1
    collect_steps_per_iteration = 100 # no of cycles
    batch_size = 500 # for training samples

    global episode_count

    actions=np.zeros([num_iterations,collect_steps_per_iteration,K*(options.M+(n_m-1))])
    rewards=np.zeros([num_iterations,collect_steps_per_iteration])

    
    def collect_step(state,environment, policy, agent_policy,buffer, gNB, iteration,step):
        time_step = environment.current_time_step()
        print("step is",step)
        P=agent_policy.action(time_step).action

        best_discrete_action = get_best_discrete_action(P)
        best_continous_action=tf.gather(agent_policy.action(time_step).action,indices=range(K),axis=1)  
        
        active_mask_cont = tf.convert_to_tensor([1.0 if UE_idx in active_UE_indices else 0.0 for UE_idx in range(K)], dtype=tf.float64)
        # Set actions for inactive UEs to zero using multiplication with the mask
        best_continous_action = best_continous_action*active_mask_cont

        action_step = policy.action(time_step) # Use the policy to choose an action based on the current time step

        if np.random.rand(1) < 0.9*(1-(iteration /num_iterations)):
            action_step = action_step.replace(action=tf.concat([best_continous_action,[valid_actions_mask[np.random.randint(0,len(valid_actions_mask)-1),:]]],1))
        else:
            action_step = action_step.replace(action=tf.concat([best_continous_action,best_discrete_action],1))

        next_time_step = environment.step(action_step.action) # Take a step in the environment using the chosen action
        options.state = next_time_step.observation # Update the global state with the new observation
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj) # Store the experience in the replay buffer for future learning
        return (traj.action,traj.reward)
     
    def collect_data(state,env, policy,agent_policy, buffer, steps, gNB,iteration):

        actions1=np.zeros([collect_steps_per_iteration,K*(options.M+(n_m-1))])
        rewards1=np.zeros([collect_steps_per_iteration])

        for step in range(0, steps):
            actions1[step,:],rewards1[step]=collect_step(state,env, policy, agent_policy,buffer, gNB,iteration,step)
        return (actions1,rewards1)
    
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agents[0].collect_data_spec, batch_size=1
    )

    reset_state()

    # This creates a TensorFlow dataset from the replay buffer for efficient training
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)
    
    # This creates an iterator for the dataset.
    iterator = iter(dataset)

    for i in range(num_iterations): #episodes

        reset_state()
        print("episode",i)
        print("episode count is",episode_count)
        print("active UEs are",active_UE_indices)

        for j in range(intermediate_iterations): #cycles
            for k in range(len(agents)):
                agent=agents[k]
                env=environments[k]
                actions[i,:,:],rewards[i,:]=collect_data(state,env, agent.collect_policy, agent.policy,replay_buffer, collect_steps_per_iteration, k, i)
            
        for k in range(len(agents)):
            agent = agents[k]
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss
            print("Loss:", train_loss)

        episode_count += 1

    # Saves actions, rewards, UE locations into .mat file
    scipy.io.savemat('git_test_Sa50.mat', mdict={'actions':actions,'rewards':rewards,'UE_locations': UE_locations})
    print("for debug")


if __name__ == "__main__":
    Fire(main)