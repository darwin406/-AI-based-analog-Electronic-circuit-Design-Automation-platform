import numpy as np
import os
from datetime import datetime
import os
import time
import matplotlib.pyplot as plt

folder_name = "simulation_log"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

if not os.path.exists('images'):
    os.makedirs('images')


settling_time_path = "simulation_log/settling_time_data.txt"
ac_analysis_path = "simulation_log/ac_analysis_data.txt"
power_path = "simulation_log/power_data.txt"
dc_sweep_path = "simulation_log/dc_sweep_data.txt"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cir_file_path = f"{folder_name}/circuitC.cir"
out_file_path = f"{folder_name}/circuitC.out"
ts_cir_file_path = f"{folder_name}/ts_circuitC.cir"
ts_out_file_path = f"{folder_name}/ts_circuitC.out"

do_plot = False
do_print = False
def get_result_from_txt(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    data = np.array([line.split() for line in lines], dtype=float)
    return data

def run_AC(W1,W2,W3,W4, VGS):  # run AC circuit   
    W5=W4
    W6=W4
    cir_file = f"""
****curciut C****

****curciut C****
.param W1=2e-07
.param W2=1.8e-07
.param W3= 3e-06
.param W4=1.8e-07
.param W5={W4}
.param W6={W4}

.param VGS = 1


V0 power GND 2
V2 Vin2 GND dc {VGS} ac 1
V1 Vin1 GND 1

*M1 Drain Gate Source Body
M1 M1G M1G power power PMOS_MODEL l=150e-9 w={W1}
M2 M2D M1G power power PMOS_MODEL l=150e-9 w={W1}
M3 M1G Vin1 M3S M3S NMOS_MODEL l=150e-9 w={W3}
M4 M2D Vin2 M3S M3S NMOS_MODEL l=150e-9 w={W3}
M5 output M2D power power PMOS_MODEL l=150e-9 w={W2}
M6 M6D M6D GND GND NMOS_MODEL l=150e-9 w={W4}
M7 M3S M6D GND GND NMOS_MODEL l=150e-9 w={W5}
M8 output M6D GND GND NMOS_MODEL l=150e-9 w={W6}

R1 power M6D 20k
R2 C1R2 output 20k
C1 M2D C1R2 0.5p
C2 output GND 1p

.model NMOS_MODEL NMOS level=54 version=4.7
.model PMOS_MODEL PMOS level=54 version=4.7

***********Analysis**********
.control
****DC Analysis****
op
let power_m1 = v(power) * @M1[id]
print power_m1
wrdata simulation_log/power_data.txt power_m1


****DC Sweep *****
dc V2 0 2 0.001
plot v(output)
wrdata simulation_log/dc_sweep_data.txt v(output)



* *****AC / Frequency response ****
ac dec 100 1 100G
plot vdb(output) 
print vdb(output)[0]
plot ph(output)*180/pi
wrdata simulation_log/ac_analysis_data.txt vdb(output) ph(output)

.endc
.end
    """

    with open(cir_file_path, 'w') as f:
        f.write(cir_file)

    os.system(f"ngspice -b {cir_file_path} -o {out_file_path} > NUL 2>&1")

def run_ts(W1,W2,W3,W4, VGS):
    W5=W4
    W6=W4
    cir_file = f"""
****Circuit C for settling time****

* SIN(offset amplitude frequency)
V0 power GND 2
* V1 Vin1 GND SIN({VGS} 0.01 1000k)
* Vin1: V-, Vin2: V+
* bind Vin1 and output
V2 Vin2 GND PULSE(0.5 1 1p 1p 1p 1u 1u)
* V1 output GND 1 

*M1 Drain Gate Source Body
M1 M1G M1G power power PMOS_MODEL l=150e-9 w={W1}
M2 M2D M1G power power PMOS_MODEL l=150e-9 w={W1}

* M3 M1G Vin1 M3S M3S NMOS_MODEL l=150e-9 w={W3}
M3 M1G output M3S M3S NMOS_MODEL l=150e-9 w={W3}

M4 M2D Vin2 M3S M3S NMOS_MODEL l=150e-9 w={W3}
M5 output M2D power power PMOS_MODEL l=150e-9 w={W2}
M6 M6D M6D GND GND NMOS_MODEL l=150e-9 w={W4}
M7 M3S M6D GND GND NMOS_MODEL l=150e-9 w={W5}
M8 output M6D GND GND NMOS_MODEL l=150e-9 w={W6}

R1 power M6D 20k
R2 C1R2 output 20k
C1 M2D C1R2 0.5p
C2 output GND 1p

.model NMOS_MODEL NMOS level=54 version=4.7
.model PMOS_MODEL PMOS level=54 version=4.7


***********Analysis**********
.control
tran 0.1n 0.5u
plot v(output) v(Vin2)
wrdata simulation_log/settling_time_data.txt v(output)
.endc
.end
"""
    with open(ts_cir_file_path, 'w') as f:
        f.write(cir_file)

    os.system(f"ngspice -b {ts_cir_file_path} -o {ts_out_file_path} > NUL 2>&1")
# ---------------process data---------------- #
def get_gain(vgs=1): # process AC data
    data = get_result_from_txt(ac_analysis_path)
    frequency = data[:,0]
    db_gain = data[:,1]
    phase = data[:,3]
    # evaluate UBW, PM, DC_gain, ts
    UBW=0
    PM=0

    DC_gain= db_gain[0]
    for i, val in enumerate(db_gain):
        if val<0:
            gbw_idx=i
            UBW = frequency[i]
            PM = phase[i]*180/np.pi
            break
    PM+=180
    while(PM>180): PM-=360
    if do_plot:
        plt.figure(figsize=(12, 8))
        plt.suptitle('AC Analysis')

        ax = plt.subplot(2, 1, 1)

        plt.plot(frequency, db_gain)

        plt.hlines(0, 1, UBW, color='g', linestyle='--', )


        plt.ylabel('Gain [dB]')
        plt.axvline(x=UBW, color='r', linestyle='--', label='UBW: {:.1e} Hz'.format(UBW))

        plt.xscale('log')
        plt.grid(alpha=0.5)
        plt.xticks(visible=False)



        plt.subplot(2, 1, 2)

        plt.plot(frequency, phase*180/np.pi)
        plt.axvline(x=UBW, color='r', linestyle='--', label='UBW: {:.1e} Hz'.format(UBW))
        plt.axhline(y=PM, color='r', linestyle='--', label='UBW: {:.1e} Hz'.format(UBW))
        plt.axhline(y=-180, color='r', linestyle='--', label='UBW: {:.1e} Hz'.format(UBW))


        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [Degree]')

        plt.xscale('log')
        plt.grid(alpha=0.5)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"images/{timestamp}_settling_time.png"
        plt.savefig(filename)
    return DC_gain, PM, UBW

def get_vgs():
    data = get_result_from_txt(dc_sweep_path)
    Vin = data[:,0]
    Vout = data[:,1]
    x = Vin
    y = Vout
    grad = abs(np.gradient(y, x))
    max_slope_index = np.argmax(grad) 
    value = grad[max_slope_index]

    vgs =  x[max_slope_index] # vgs
    vgs = round(vgs, 4)
    return vgs # vgs

def get_ts():
    data = get_result_from_txt(settling_time_path)
    time = data[:,0]
    output = data[:,1]

    final_value = output[-1]
    tolerance_band = 0.01 * final_value

    # Find the time when the output enters the tolerance band and stays there
    for t, v in zip(time, output):
        if abs(v - final_value) <= tolerance_band:
            settling_time = round(t,10)
            break
    tolerance_value = 0.01

    if(do_plot):
 
        plt.figure(figsize=(10, 6))

        plt.plot(time, output, label='Output Voltage', color='red')


        plt.axhline(y=final_value, color='green', linestyle='--', label='Final Value')
        plt.axhline(y=final_value * (1 + tolerance_value), color='blue', linestyle=':', label='+1% Tolerance')
        plt.axhline(y=final_value * (1 - tolerance_value), color='blue', linestyle=':', label='-1% Tolerance')

        if settling_time is not None:
            plt.axvline(x=settling_time, color='purple', linestyle='--', label=f'Settling Time: {settling_time:.6e} s')

        plt.title(f'Settling Time Verification')
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.legend()

        plt.grid(True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"images/{timestamp}_settling_time.png"
        plt.savefig(filename)
        print(f"plot saved as {filename}")
    
    return settling_time

def simulate(w1,w2,w3,w4, enable_plot= False, enable_print=False):
    start_time = time.time()
    global do_print
    global do_plot
    do_print = enable_print
    do_plot = enable_plot

    run_AC(w1,w2,w3,w4, 1)
    vgs_opt = get_vgs()

    run_AC(w1,w2,w3,w4, vgs_opt)
    gain, PM, UBW = get_gain() #dbGain
    power = float(get_result_from_txt(power_path)[:,1])

    run_ts(w1,w2,w3,w4, vgs_opt)
    ts=get_ts()
    FOM_val = cal_FOM(w1,w2,w3,w4,PM, UBW, power, ts, gain)
    end_time = time.time()

    elapsed_time = end_time - start_time

    if(do_print):
        print("="*20+"result"+"="*20)
        print(f"[{w1}, {w2}, {w3}, {w4}]")
        print(f"vgs: {vgs_opt} [V]")
        if(gain != None):
            print(f"DC gain: {gain} [dB]")
            print(f"Phase Margin: {PM} [degree]")
            print("Unity-gain bandwidth: %.5e [Hz]"%UBW)
        else:
            print("AC analysis failed")
        print("Power: %.5e [W]" %power)
        print(f"1% settling time: {ts} [s]")
        print(f"FOM: {FOM_val}")
        print(f"Program executed in {elapsed_time:.2f} seconds")

    return FOM_val

arr = [[] for _ in range(6)]
def cal_FOM(w1,w2,w3,w4, PM, UBW, power, ts, DC_gain):
    
    r1 = DC_gain /70
    r2 = np.log10(UBW)/9  
    r3 = np.log10(power) /-3
    r4 = (w1+w2+w3+w4) *180 *500
    
    r5 = np.log10(ts)/-9


    result = r1+r2-r3-r4-r5
    if PM>=60:
        pass
    else:
        result-=1 # fixed it 100 to 1
    return result


if __name__ == "__main__":
    simulate(807.5781154632568e-9, 2143.546466231346e-9, 2142.2211009263992e-9, 243.2319825142622e-9, enable_plot=False, enable_print=True)

    # Next state:
    simulate(1480.5863285064697e-9, 787.5345253944397e-9, 2250.0e-9, 180.0e-9, enable_plot=False, enable_print=True)