import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from qm.qua import *
from qm.simulate import SimulationConfig
from configuration import config
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
from qm import LoopbackInterface

# method 1: ensemble average: frequency sweep and demodulation
f_min = 42e6; f_max= 58e6; df = 0.05e6; f_vec = np.arange(f_min, f_max, df)

with program() as PSD_method1:

    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    P_st = declare_stream()

    update_frequency('drive_element_1', 48e6)
    update_frequency('drive_element_2', 52e6)
    with for_(f, f_min, f < f_max, f+df):
        update_frequency('readout_element', f)
        align()
        play('const', 'drive_element_1')
        play('const', 'drive_element_2')
        measure('readout'*amp(0), 'readout_element', None, dual_demod.full('Wc', 'out1', 'Ws', 'out2', I),
                dual_demod.full('-Ws', 'out1', 'Wc', 'out2', Q))
        save(I, I_st)
        save(Q, Q_st)

    with stream_processing():
        (I_st*I_st + Q_st*Q_st).save_all('PSD')


qmm = QuantumMachinesManager()
job = qmm.simulate(config, PSD_method1, simulate=SimulationConfig(duration=120000,
                                                          simulation_interface=LoopbackInterface([
                                                              ('con1', 1, 'con1', 1),
                                                              ('con1', 2, 'con1', 2)])))

job.result_handles.wait_for_all_values()
PSD = job.result_handles.get('PSD').fetch_all()['value']
plt.plot(f_vec, PSD)
