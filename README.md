# AI-based-analog-EDA-platform
 This is software that automatically optimizes transistor size using reinforcement learning. <br/>
 <br/>
 Optimization of analog electronic circuits has traditionally been considered an expensive task that requires <br/>
 a large investment of time and effort by skilled engineers because it is difficult to automate with software.<br/>
 Research is continuing to reduce the cost, time, and performance of analog electronic circuit design <br/>
 by enabling automation through reinforcement learning. <br/>
 <br/>
 In this project, we show that reinforcement learning on a simple circuit can be used to find the optimal FOM <br/>
 and its corresponding state in fewer simulations than random search, grid search, or hand computation.<br/>
 By improving and modifying the methods proposed in DDPG(https://arxiv.org/pdf/1509.02971) and DNN_OPT(https://arxiv.org/abs/2110.00211) papers, we succeeded in optimizing the transistor size. <br/>
 <br/>

## Circuit
 We experimented with a total of three simple circuits and found the optimal FOM and state. <br/>
 The code we've uploaded to GitHub is for one of the three circuits, which we'll call circuit C. <br/>
 Circuit C is shown in the photo below. <br/>
 ![circuit_C](https://github.com/user-attachments/assets/696361a2-0978-416c-8579-64661d5dfd8a) <br/>
 In this circuit, W<sub>1</sub>,W<sub>2</sub>,W<sub>3</sub>,W<sub>4</sub> are optimized using reinforcement learning. <br/>

## Simulator
NgSpice

## FOM
FOM was set and reinforcement learning was performed to maximize or minimize the FOM according to transistor sizes. <br/>
In circuit C, the optimization is to maximize the FOM. <br/>
We set up the FOM as follows. <br/>

$$ FOM = \frac{GAIN(dB)}{70(dB)} + \frac{\log_{10}(GBW)}{\log_{10}(1GHz)} - \frac{\log_{10}(Power)}{\log_{10}(1mW)}- (w1 + w2 + w3 + w4) * 180 * 500- \frac{\log_{10}(settling \ time)}{\log_{10}(1ns)} \quad \text{subject to } PM \geq 60^\circ $$

*PM: Phase Margin* <br/>
*GBW: Gain Bandwidth*

## Environment setting 
$$Normalization = \frac{x - x_{min}}{x_{max} - x_{min}}$$ <br/>
*x : Actual transistor width* <br/>
Minimum transistor width : 180nm <br/>
Maximum transistor width : 2250nm <br/>
<br/>
Observation(state) space: **W** = (w₁, w₂, w₃, w₄), wᵢ ∈ [0, 1] for all i ∈ {1, 2, 3, 4} <br/>
<br/>
**Reinforcement learning is done in observation space by normalizing the actual transistor width size.** <br/>
<br/>
Action space: **A** = (Δw₁, Δw₂, Δw₃, Δw₄) <br/>
In Algorithm 1, Δwᵢ ∈ [-0.1,0.1] for all i ∈ {1, 2, 3, 4} <br/>
In Algorithm 2, Δwᵢ ∈ [-1,1] for all i ∈ {1, 2, 3, 4} <br/>

## Algorithm 1: DDPG (Action scale = 0.1)
 It is almost identical to the commonly known DDPG algorithm. However, it does not use a target network. <br/>
 This is the DDPG-based reinforcement learning algorithm we implemented. <br/>
 ![ddpg_algoritm](https://github.com/user-attachments/assets/e9cf110a-a866-43be-ada2-4d9ba8e21f04) <br/>
 For the neural network, we used MLP. <br/>
## Algorithm 2: DDPG (Action scale = 1)
 This algorithm uses target networks. <br/>
 There are two target networks: Actor Target Network and Critic Target Network. <br/>
## Algorithm 3: DNN_OPT (Action scale = 1)
 This is the DNN OPT-based reinforcement learning algorithm we implemented. <br/>
 ![dnn_opt_fix](https://github.com/user-attachments/assets/8d824afb-faf8-47fe-a874-5bc934f7dbb7) <br/>
 <br/>
 The formulas involved are as follows.<br/>
 ![ean1_fix](https://github.com/user-attachments/assets/d4f773bd-6637-4757-a1b9-afb8d61f19da) <br/>
 ![eqn_fix](https://github.com/user-attachments/assets/eacb5a33-8ee7-4fb1-a492-c5e17b043d30) <br/>
 For the neural network, we used MLP. <br/>
 
## Result
### Algorithm 1
![algorithm1_result](https://github.com/user-attachments/assets/b803a8eb-7bf8-4fc3-aa88-d04e280ca245) <br/>
This graph is for results tested after training is complete. <br/>
We can see that it trained well in terms of what we know about reinforcement learning. <br/>
### Algorithm 2
![algorithm_2_result](https://github.com/user-attachments/assets/e950cda1-3c95-4bbe-81e0-937dcd7c6760) <br/>
This graph shows the highest FOM values generated in the replay memory, depending on the step in the training process. <br/>
Since our goal is to get the state with the highest FOM, we don't necessarily need to test it. <br/>
### Algorithm 3
![algoritm3_result](https://github.com/user-attachments/assets/7c852e40-0b9f-4e10-920a-a895681cfd6a) <br/>
This graph shows the highest FOM values generated in the replay memory, depending on the step in the training process. <br/>

### Comparison by algorithm
| Algorithm Number                                 | Number of trainings      | Number of simulations | Highest FOM reached |
|-------------------------------------------|---------------|----------------|-----------------|
| 1 | 4000          | 8000           | -0.9516993      |
| 2  | 750           | 1500           | -0.9495         |
| 3 | 119 | 238          | -0.9498         |
