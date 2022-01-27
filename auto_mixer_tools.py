# This file contains classes of spectrum analyzers using the VISA interface to communicate with the computers.
# They should have almost uniform commands, making adaptions to new models/brands quite easy

from qm.qua import *
from abc import ABC, abstractmethod
import numpy as np
import pyvisa as visa
import time
import matplotlib.pyplot as plt
import scipy.optimize as opti
from qm.QuantumMachinesManager import QuantumMachinesManager

def auto_mixer_cal(element, pulse, LO, IF, config, qm):
    address = (
    "TCPIP::192.168.1.151::INSTR")  # The address for the SA, opened using visa.
    
    bDoSweeps = True  # If True, performs a large sweep before and after the optimization.
    method = 1  # If set to 1, checks power using a channel power measurement. If set to 2, checks power using a marker.

    # Parameters for SA - Measurement:
    measBW = 100  # Measurement bandwidth
    measNumPoints = 101

    # Parameters for SA - Sweep:
    sweepBW = 1e3
    fullNumPoints = 1201
    fullSpan = int(abs(IF * 4.1))  # Larger than 4 such that we'll see spurs
    startFreq = LO - fullSpan / 2
    stopFreq = LO + fullSpan / 2
    freq_vec = np.linspace(float(startFreq), float(stopFreq), int(fullNumPoints))

    # Parameters for Nelder-Mead
    initial_simplex = np.zeros([3, 2])
    initial_simplex[0, :] = [0, 0]
    initial_simplex[1, :] = [0, 0.1]
    initial_simplex[2, :] = [0.1, 0]
    xatol = 1e-4  # 1e-4 change in DC offset or gain/phase
    fatol = 3  # dB change tolerance
    maxiter = 50  # 50 iterations should be more then enough, but can be changed.
    
    ##########
    # Execute:
    ##########

    # Execute the mixer_cal program:
    #qmm = QuantumMachinesManager()
    #qm = qmm.open_qm(config)

    calib = KeysightFieldFox(address, qm, pulse, element)
    calib.method = method
    calib.set_mode_SA()

    # Set video bandwidth to be automatic and bandwidth to be manual. Disable continuous mode
    calib.set_automatic_video_bandwidth(1)
    calib.set_automatic_bandwidth(0)
    calib.set_cont_off()
    
    if bDoSweeps:
        # Set Bandwidth and start/stop freq of SA for a large sweep
        calib.set_bandwidth(sweepBW)
        calib.set_sweep_points(fullNumPoints)
        calib.set_center_freq(LO)
        calib.set_span(fullSpan)

        # Do a single read
        calib.get_single_trigger()

        # Query the FieldFox response data
        amp1 = calib.get_full_trace()

    # Configure measure
    if method == 1:  # Channel power
        calib.enable_measurement()
        calib.sets_measurement_integration_bw(10 * measBW)
        calib.disables_measurement_averaging()
    elif method == 2:  # Marker
        calib.get_single_trigger()
        calib.active_marker(1)
    calib.set_sweep_points(measNumPoints)
    calib.set_span(14 * measBW)
    calib.set_bandwidth(measBW)

    # Get Signal
    calib.set_center_freq(LO + IF)
    if method == 2:  # Marker
        calib.set_marker_freq(1, LO + IF)
    signal = int(calib.get_amp())

    # Optimize LO leakage
    calib.set_center_freq(LO)
    if method == 2:  # Marker
        calib.set_marker_freq(1, LO)
    start_time = time.time()
    fun_leakage = lambda x: calib.get_leakage(element, x[0], x[1]) 
    res_leakage = opti.minimize(
        fun_leakage,
        [0, 0],
        method="Nelder-Mead",
        options={
            "xatol": xatol,
            "fatol": fatol,
            "initial_simplex": initial_simplex,
            "maxiter": maxiter,
        },
    )
    print(
        f"LO Leakage Results: Found a minimum of {int(res_leakage.fun)} dBm at I0 = {res_leakage.x[0]:.5f}, Q0 = {res_leakage.x[1]:.5f} in "
        f"{int(time.time() - start_time)} seconds --- {signal - int(res_leakage.fun)} dBc"
    )

    # Optimize image
    calib.set_center_freq(LO - IF)
    if method == 2:  # Marker
        calib.set_marker_freq(1, LO - IF)
    start_time = time.time()
    fun_image = lambda x: calib.get_image(element, x[0], x[1]) 
    res_image = opti.minimize(
        fun_image,
        [0, 0],
        method="Nelder-Mead",
        options={
            "xatol": xatol,
            "fatol": fatol,
            "initial_simplex": initial_simplex,
            "maxiter": maxiter,
        },
    )
    print(
        f"Image Rejection Results: Found a minimum of {int(res_image.fun)} dBm at g = {res_image.x[0]:.5f}, phi = {res_image.x[1]:.5f} in "
        f"{int(time.time() - start_time)} seconds --- {signal - int(res_image.fun)} dBc"
    )

    # Turn measurement off
    if method == 1:  # Channel power
        calib.disables_measurement()

    # Set parameters back for a large sweep
    if bDoSweeps:
        # Set Bandwidth and start/stop freq of SA for a large sweep
        calib.set_bandwidth(sweepBW)
        calib.set_sweep_points(fullNumPoints)
        calib.set_center_freq(LO)
        calib.set_span(fullSpan)

        # Do a single read
        calib.get_single_trigger()

        # Query the FieldFox response data
        amp2 = calib.get_full_trace()

        #plt.figure("Full Spectrum")
        plt.figure(figsize=(8,6), dpi = 100)
        plt.plot(freq_vec, amp1, label = 'before')
        plt.plot(freq_vec, amp2, label = 'after')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Spectrum (dBm)')
        plt.xlim(freq_vec[0],freq_vec[-1])
        plt.legend()
        plt.show()

    # Return the FieldFox back to continuous mode
    calib.set_cont_on()

    # On exit clean a few items up.
    #calib.__del__()
    
    
class VisaSA(ABC):
    def __init__(self, address, qm, pulse, element):
        # Gets an existing qm, assumes there is an element called "qubit" with an operation named "test_pulse" which
        # plays a constant pulse
        super().__init__()
        rm = visa.ResourceManager()
        self.sa = rm.open_resource(address)
        self.sa.timeout = 100000

        with program() as mixer_cal:
            with infinite_loop_():
                play(pulse, element)

        self.qm = qm
        self.job = qm.execute(mixer_cal)
        self.method = None

    def IQ_imbalance_correction(self, g, phi):
        c = np.cos(phi)
        s = np.sin(phi)
        N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
        return [
            float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]
        ]

    def get_leakage(self, element, i0, q0):
        self.qm.set_dc_offset_by_qe(element, "I", i0)
        self.qm.set_dc_offset_by_qe(element, "Q", q0)
        amp_ = self.get_amp()
        return amp_

    def get_image(self, element, g, p):
        self.job.set_element_correction(element, self.IQ_imbalance_correction(g, p))
        amp_ = self.get_amp()
        return amp_

    def __del__(self):
        self.sa.clear()
        self.sa.close()

    @abstractmethod
    def get_amp(self):
        pass

    @abstractmethod
    def set_automatic_video_bandwidth(self, state: int):
        # State should be 1 or 0
        pass

    @abstractmethod
    def set_automatic_bandwidth(self, state: int):
        # State should be 1 or 0
        pass

    @abstractmethod
    def set_bandwidth(self, bw: int):
        # Sets the bandwidth
        pass

    @abstractmethod
    def set_sweep_points(self, n_points: int):
        # Sets the number of points for a sweep
        pass

    @abstractmethod
    def set_center_freq(self, freq: int):
        # Sets the central frequency
        pass

    @abstractmethod
    def set_span(self, span: int):
        # Sets the span
        pass

    @abstractmethod
    def set_cont_off(self):
        # Sets continuous mode off
        pass

    @abstractmethod
    def set_cont_on(self):
        # Sets continuous mode on
        pass

    @abstractmethod
    def get_single_trigger(self):
        # Performs a single sweep
        pass

    @abstractmethod
    def active_marker(self, marker: int):
        # Active the given marker
        pass

    @abstractmethod
    def set_marker_freq(self, marker: int, freq: int):
        # Sets the marker's frequency
        pass

    @abstractmethod
    def query_marker(self, marker: int):
        # Query the marker
        pass

    @abstractmethod
    def get_full_trace(self):
        # Returns the full trace
        pass

    @abstractmethod
    def enable_measurement(self):
        # Sets the measurement to channel power
        pass

    @abstractmethod
    def disables_measurement(self):
        # Sets the measurement to none
        pass

    @abstractmethod
    def sets_measurement_integration_bw(self, ibw: int):
        # Sets the measurement integration bandwidth
        pass

    @abstractmethod
    def disables_measurement_averaging(self):
        # Disables averaging in the measurement
        pass

    @abstractmethod
    def get_measurement_data(self):
        # Returns the result of the measurement
        pass

class RohdeSchwarzFPC1000(VisaSA):
    def get_amp(self):
        self.get_single_trigger()
        if self.method == 1:  # Channel power
            sig = self.get_measurement_data()
        elif self.method == 2:  # Marker
            sig = self.query_marker(1)
        else:
            sig = float("NaN")
        return sig

    def set_automatic_video_bandwidth(self, state: int):
        # State should be 1 or 0
        self.sa.write(f"SENS:BAND:VID:AUTO {int(state)}")

    def set_automatic_bandwidth(self, state: int):
        # State should be 1 or 0. Resolution (or measurement) bandwidth
        self.sa.write(f"SENS:BAND:AUTO {int(state)}")

    def set_bandwidth(self, bw: int):
        # Sets the resolution (or measurement) bandwidth, 1 Hz to 3 MHz, default unit is Hz
        # Example SENS:BAND 100000
        self.sa.write(f"SENS:BAND {int(bw)}")

    def set_sweep_points(self, n_points: int):
        # Sets the number of points for a sweep, allowed range 101 to 2501, default is 201
        self.sa.write(f"SENS:SWE:POIN {int(n_points)}")

    def set_center_freq(self, freq: int):
        # Sets the central frequency, default unit is Hz
        self.sa.write(f"SENS:FREQ:CENT {int(freq)}")

    def set_span(self, span: int):
        # Sets the span, default unit is Hz
        self.sa.write(f"SENS:FREQ:SPAN {int(span)}")

    def set_cont_off(self):
        # This command selects the sweep mode (but does not start the measurement!)
        # OFF or 0 is a single sweep mode
        # *OPC? is to make sure there is no overlapping execution
        return self.sa.query("INIT:CONT OFF;*OPC?")

    def set_cont_on(self):
        # This command selects the sweep mode (but does not start the measurement!)
        # ON or 1 is a continuous sweep mode
        # *OPC? is to make sure there is no overlapping execution
        return self.sa.query("INIT:CONT ON;*OPC?")

    def get_single_trigger(self):
        # Initiates a new measurement sequence (starts the sweep)
        return self.sa.query("INIT:IMM;*OPC?")

    def active_marker(self, marker: int):
        # Activate the given marker
        self.sa.write(f"CALC:MARK{int(marker)} ON")

    def set_marker_freq(self, marker: int, freq: int):
        # Sets the marker's frequency. Default unit is Hz
        self.get_single_trigger()
        self.sa.write(f"CALC:MARK{int(marker)}:X {int(freq)}")

    def query_marker(self, marker: int):
        # Query the amplitude (default unit is dBm) of the marker
        return float(self.sa.query(f"CALC:MARK{int(marker)}:Y?"))

    def get_full_trace(self):
        # Returns the full trace. Implicit assumption that this is trace1 (there could be 1-4)
        self.sa.write("FORM ASC")  # data format needs to be in ASCII
        ff_SA_Trace_Data = self.sa.query("TRAC:DATA? TRACE1")
        # Data from the FPC comes out as a string of 1183 values separated by ',':
        # '-1.97854112E+01,-3.97854112E+01,-2.97454112E+01,-4.92543112E+01,-5.17254112E+01,-1.91254112E+01...\n'
        # The code below turns it into an a python list of floats
        # Use split to turn long string to an array of values
        ff_SA_Trace_Data_Array = ff_SA_Trace_Data.split(",")
        amp = [float(i) for i in ff_SA_Trace_Data_Array]
        return amp

    def enable_measurement(self):
        # Sets the measurement to channel power
        self.sa.write(
            "CALC:MARK:FUNC:POW:SEL CPOW; CALC:MARK:FUNC:LEV:ONCE; CALC:MARK:FUNC:CPOW:UNIT DBM; CALC:MARK:FUNC:POW:RES:PHZ ON"
        )

    def disables_measurement(self):
        # Sets the channel power measurement to none
        self.sa.write("CALC:MARK:FUNC:POW OFF")

    def sets_measurement_integration_bw(self, ibw: int):
        # Sets the measurement integration bandwidth for channel power measurements
        self.sa.write(f"CALC:MARK:FUNC:CPOW:BAND {int(ibw)}")

    def disables_measurement_averaging(self):
        # disables averaging in the measurement
        pass

    def get_measurement_data(self):
        # Returns the result of the measurement
        return self.sa.query(f"CALC:MARK:FUNC:POW:RES? CPOW")


class KeysightFieldFox(VisaSA):
    def get_amp(self):
        self.get_single_trigger()
        if self.method == 1:  # Channel power
            sig = self.get_measurement_data()
        elif self.method == 2:  # Marker
            sig = self.query_marker(1)
        else:
            sig = float("NaN")
        return sig

    def set_mode_SA(self):
        self.sa.write("INSTrument:SELect 'SA'")
    
    def set_automatic_video_bandwidth(self, state: int):
        # State should be 1 or 0
        #self.sa.write(f"SENS:BAND:VID:AUTO {int(state)}")
        self.sa.write(f":SENSe:BANDwidth:VIDeo:AUTO {int(state)}")
        #[:SENSe]:BANDwidth:VIDeo:AUTO <bool>
        
    def set_automatic_bandwidth(self, state: int):
        # State should be 1 or 0
        self.sa.write(f"SENS:BAND:AUTO {int(state)}")

    def set_bandwidth(self, bw: int):
        # Sets the bandwidth
        self.sa.write(f":SENS:BAND {int(bw)}")
        #self.sa.write(f"SENSe:BWIDth {int(bw)}")

    def set_sweep_points(self, n_points: int):
        # Sets the number of points for a sweep
        self.sa.write(f"SENS:SWE:POIN {int(n_points)}")

    def set_center_freq(self, freq: int):
        # Sets the central frequency
        self.sa.write(f"SENS:FREQ:CENT {int(freq)}")

    def set_span(self, span: int):
        # Sets the span
        self.sa.write(f"SENS:FREQ:SPAN {int(span)}")

    def set_cont_off(self):
        return self.sa.query("INIT:CONT OFF;*OPC?")

    def set_cont_on(self):
        # Sets continuous mode on
        return self.sa.query("INIT:CONT ON;*OPC?")

    def get_single_trigger(self):
        # Performs a single sweep
        return self.sa.query("INIT:IMM;*OPC?")

    def active_marker(self, marker: int):
        # Active the given marker
        self.sa.write(f"CALC:MARK{int(marker)}:ACT")

    def set_marker_freq(self, marker: int, freq: int):
        # Sets the marker's frequency
        self.get_single_trigger()
        self.sa.write(f"CALC:MARK{int(marker)}:X {int(freq)}")

    def query_marker(self, marker: int):
        # Query the marker
        return float(self.sa.query(f"CALC:MARK{int(marker)}:Y?"))

    def get_full_trace(self):
        # Returns the full trace
        ff_SA_Trace_Data = self.sa.query("TRACE:DATA?")
        # Data from the Fieldfox comes out as a string separated by ',':
        # '-1.97854112E+01,-3.97854112E+01,-2.97454112E+01,-4.92543112E+01,-5.17254112E+01,-1.91254112E+01...\n'
        # The code below turns it into an a python list of floats

        # Use split to turn long string to an array of values
        ff_SA_Trace_Data_Array = ff_SA_Trace_Data.split(",")
        amp = [float(i) for i in ff_SA_Trace_Data_Array]
        return amp

    def enable_measurement(self):
        # Sets the measurement to channel power
        self.sa.write("SENS:MEAS:CHAN CHP")

    def disables_measurement(self):
        # Sets the measurement to none
        self.sa.write("SENS:MEAS:CHAN NONE")

    def sets_measurement_integration_bw(self, ibw: int):
        # Sets the measurement integration bandwidth
        self.sa.write(f"SENS:CME:IBW {int(ibw)}")

    def disables_measurement_averaging(self):
        # disables averaging in the measurement
        self.sa.write("SENS:CME:AVER:ENAB 0")

    def get_measurement_data(self):
        # Returns the result of the measurement
        return float(self.sa.query("CALC:MEAS:DATA?").split(",")[0])
        # Data from the Fieldfox comes out as a string separated by ',':
        # '-1.97854112E+01,-3.97854112E+01\n'
        # The code above takes the first value and converts to float.


class KeysightXSeries(VisaSA):
    def get_amp(self):
        self.get_single_trigger()
        if self.method == 1:  # Channel power
            sig = self.get_measurement_data()
        elif self.method == 2:  # Marker
            sig = self.query_marker(1)
        else:
            sig = float("NaN")
        return sig

    def set_automatic_video_bandwidth(self, state: int):
        # State should be 1 or 0
        self.sa.write(f"SENS:BAND:VID:AUTO {int(state)}")

    def set_automatic_bandwidth(self, state: int):
        # State should be 1 or 0
        self.sa.write(f"SENS:BAND:AUTO {int(state)}")

    def set_bandwidth(self, bw: int):
        # Sets the bandwidth
        self.sa.write(f"SENS:BAND {int(bw)}")
        #self.sa.write(f"SENSe:BAND {int(bw)}")

    def set_sweep_points(self, n_points: int):
        # Sets the number of points for a sweep
        self.sa.write(f"SENS:SWE:POIN {int(n_points)}")

    def set_center_freq(self, freq: int):
        # Sets the central frequency
        self.sa.write(f"SENS:FREQ:CENT {int(freq)}")

    def set_span(self, span: int):
        # Sets the span
        self.sa.write(f"SENS:FREQ:SPAN {int(span)}")

    def set_cont_off(self):
        return self.sa.query("INIT:CONT OFF;*OPC?")

    def set_cont_on(self):
        # Sets continuous mode on
        return self.sa.query("INIT:CONT ON;*OPC?")

    def get_single_trigger(self):
        # Performs a single sweep
        return self.sa.query("INIT:IMM;*OPC?")

    def active_marker(self, marker: int):
        # Active the given marker
        self.sa.write(f"CALC:MARK{int(marker)}:MODE POS")

    def set_marker_freq(self, marker: int, freq: int):
        # Sets the marker's frequency
        self.get_single_trigger()
        self.sa.write(f"CALC:MARK{int(marker)}:X {int(freq)}")

    def query_marker(self, marker: int):
        # Query the marker
        return float(self.sa.query(f"CALC:MARK{int(marker)}:Y?"))

    def get_full_trace(self):
        # Returns the full trace
        ff_SA_Trace_Data = self.sa.query("TRACE:DATA? TRACE1")
        # Data from the Keysight comes out as a string separated by ',':
        # '-1.97854112E+01,-3.97854112E+01,-2.97454112E+01,-4.92543112E+01,-5.17254112E+01,-1.91254112E+01...\n'
        # The code below turns it into an a python list of floats

        # Use split to turn long string to an array of values
        ff_SA_Trace_Data_Array = ff_SA_Trace_Data.split(",")
        amp = [float(i) for i in ff_SA_Trace_Data_Array]
        return amp

    def enable_measurement(self):
        # Sets the measurement to channel power
        self.sa.write(":CONF:CHP")

    def disables_measurement(self):
        # Sets the measurement to none
        self.sa.write(":CONF:CHP NONE")

    def sets_measurement_integration_bw(self, ibw: int):
        # Sets the measurement integration bandwidth
        self.sa.write(f"SENS:CHP:BAND:INT {int(ibw)}")

    def disables_measurement_averaging(self):
        # disables averaging in the measurement
        self.sa.write("SENS:CHP:AVER 0")

    def get_measurement_data(self):
        # Returns the result of the measurement
        return float(self.sa.query("READ:CHP?").split(",")[0])
        # Data from the Keysight comes out as a string separated by ',':
        # '-1.97854112E+01,-3.97854112E+01\n'
        # The code above takes the first value and converts to float.