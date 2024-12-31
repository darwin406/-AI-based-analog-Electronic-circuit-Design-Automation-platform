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

## FOM
FOM was set and reinforcement learning was performed to maximize or minimize the FOM according to transistor sizes. <br/>
In circuit C, the optimization is to maximize the FOM. <br/>
We set up the FOM as follows. <br/>

$$ FOM = \frac{GAIN(dB)}{70(dB)} + \frac{\log_{10}(GBW)}{\log_{10}(1GHz)} - \frac{\log_{10}(Power)}{\log_{10}(1mW)}- (w1 + w2 + w3 + w4) * 180 * 500- \frac{\log_{10}(settling \ time)}{\log_{10}(1ns)} \quad \text{subject to } PM \geq 60^\circ $$

*PM: Phase Margin* <br/>
*GBW: Gain Bandwidth*

## Environment setting



## Algorithm 1: DDPG (Action scale = 0.1)
 It is almost identical to the commonly known DDPG algorithm. However, it does not use a target network. <br/>
 


## Algorithm 2: DNN_OPT (Action scale = 1)
 

 
 
