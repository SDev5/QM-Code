from scipy.signal import gaussian
import numpy as np

def IQ_imbalance_correction(g, phi):
        c = np.cos(phi)
        s = np.sin(phi)
        N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
        return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


#######################
# LO frequencies [Hz] #
#######################
RO_lo = 4.26435e9
Q1_lo = 5.84755e9
#Q2_lo=  6.100e9
#diff_lo = 1.2e9
#sum_lo = 13e9

#######################
# IF frequencies [Hz] #
#######################
RR_1_IF = 25e6 #44.45e6, 47.2e6
Q1_IF = 20e6
#Q1_XY_ef_IF = -218e6
#Q2_IF = 88.28e6
#RR_2_IF = -230.132e6#228.95e6#-229.35e6
#flux_L_IF = 5e6
#flux_R_IF = 5e6
#diff_IF = 70e6
#sum_IF = 51e6

########################
# pulses's length [ns] #
########################

cw_len = 600
gauss_len = 20
# Q2_pi_len = 40
# Q2_pi2_len = 40
Q1_ef_pi_len = 40
Q1_pi_len = 96
Q1_pi2_len = 68 #56 #40
sat_Q1_len = 5000 #152
# sat_Q2_len = 40000
readout_len = 152 #1000  
long_ro_len = 2000

#######################
# Pulse durations (QUA Program)
#######################
t_Q1_pi = 12

# Pulse amps 
amp_Q1_pi = 1.55

########################
# mixer configs #
########################

#R&S IQ settings

#order of element in array: I Offset, Q Offset, Gain Imbalance, Quadrature Offset
R1_RS = ['0.86','-0.86','0.025','0.00']
Q1_RS = ['-2.04','-0.02','0.0','0.12']

#RR1 
R1_corr_mat = IQ_imbalance_correction(0.02749, -0.05493) #[1.0090486503481335, 0.13688252842553544, 0.13230631803569456, 1.0439496209603782]
R1_corr_I = 0.00307 #0.00645    
R1_corr_Q = 0.00375 #-0.00596 
R1_offset_I= +0.023541468248869244 + 0.0001424852873149671
#R1_offset_Q= -0.2582120138183594

#Q1
Q1_corr_mat=IQ_imbalance_correction(-0.01279, -0.13849)
Q1_corr_I= -0.01107
Q1_corr_Q= 0.00251

########################
# THE QM CONFIGURATION #
########################

config = {

    'version': 1,

    'controllers': {

        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': R1_corr_I},   # Multiplexed_RO_I
                2: {'offset': R1_corr_Q},   # Multiplexed_RO_Q
                3: {'offset': Q1_corr_I},   
                4: {'offset': Q1_corr_Q},  
                # 5: {'offset': 0.0},   # Q1_z
                # 6: {'offset': 0.0},   # Q2_z
                # 7: {'offset': 0.0},   # Coupler diff I
                # 8: {'offset': 0.0},   # Coupler diff Q
                # 9: {'offset': 0.0},   # Coupler sum I
                # 10: {'offset': 0.0},  # Coupler sum Q

            },
            'digital_outputs': {},
            'analog_inputs': {
                1: {'offset': R1_offset_I, 'gain_db':3},  # Multiplexed_RO_I
                #2: {'offset': R1_offset_Q, 'gain_db':3},  # Multiplexed_RO_Q

            }
        },
    },

    'elements': {

        'RR_1': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': RO_lo,
                'mixer': 'mixer_RO',
            },
            'intermediate_frequency': RR_1_IF,
            'operations': {
                'const': 'const_pulse',
                'readout': 'readout_pulse_in',
                'long_readout': 'long_readout_pulse_in',
                'cw_readout': 'cw_readout',
            },
            'outputs': {
                'out1': ('con1', 1),
                #'out2': ('con1', 2),
            },
            'time_of_flight': 320, 
            'smearing': 0, 
        },

        # 'RR_2': {
        #     'mixInputs': {
        #         'I': ('con1', 1),
        #         'Q': ('con1', 2),
        #         'lo_frequency': RO_lo,
        #         'mixer': 'mixer_RO',
        #     },
        #     'intermediate_frequency': RR_2_IF,
        #     'operations': {
        #         'const': 'const_pulse',
        #         'readout': 'readout_pulse_Q2',
        #         'long_readout': 'long_readout_pulse_Q2',
        #         'cw_readout': 'cw_readout',
        #     },
        #     'outputs': {
        #         'out1': ('con1', 1),
        #         'out2': ('con1', 2),
        #     },
        #     'time_of_flight': 260,
        #     'smearing': 0,
        # },

        'Q1': {
            'mixInputs': {
                'I': ('con1', 3),
                'Q': ('con1', 4),
                'lo_frequency': Q1_lo,
                'mixer': 'mixer_Q1',
            },
            'intermediate_frequency': Q1_IF,
            'operations': {
                'const': 'const_pulse',
                'saturation': 'saturation_pulse_Q1',
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse_Q1',
                'pi2': 'pi_2_pulse_Q1',
            },
        },

    #     'Q2_xy': {
    #         'mixInputs': {
    #             'I': ('con1', 3),
    #             'Q': ('con1', 4),
    #             'lo_frequency': XY2_lo,
    #             'mixer': 'mixer_XY',
    #         },
    #         'intermediate_frequency': Q2_XY_IF,
    #         'operations': {
    #             'const': 'const_pulse',
    #             'saturation': 'saturation_pulse_Q2',
    #             'gaussian': 'gaussian_pulse',
    #             'pi': 'pi_pulse_Q2',
    #             'pi2': 'pi_2_pulse_Q2',
    #         },
    #     },

    #     'Q1_z': {
    #         'singleInput': {
    #             'port': ('con1', 5)
    #         },
    #         'intermediate_frequency': flux_L_IF,
    #         'operations': {}
    #     },

    #     'Q2_z': {
    #         'singleInput': {
    #             'port': ('con1', 6)
    #         },
    #         'intermediate_frequency': flux_R_IF,
    #         'operations': {}
    #     },

    #     'Coupler_diff': {
    #         'mixInputs': {
    #             'I': ('con1', 7),
    #             'Q': ('con1', 8),
    #             'lo_frequency': diff_lo,
    #             'mixer': 'mixer_diff',
    #         },
    #         'intermediate_frequency': diff_IF,
    #         'operations': {}
    #     },

    #     'Coupler_sum': {
    #         'mixInputs': {
    #             'I': ('con1', 9),
    #             'Q': ('con1', 10),
    #             'lo_frequency': sum_lo,
    #             'mixer': 'mixer_sum',
    #         },
    #         'intermediate_frequency': sum_IF,
    #         'operations': {}
    #     },

    },

    'pulses': {

        #################
        # common pulses #
        #################

        'const_pulse': {
            'operation': 'control',
            'length': 100,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'const_wf', #'zero_wf',
            },
        },

        'gaussian_pulse': {
            'operation': 'control',
            'length': gauss_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf',
            },
        },
        
        'cw_readout': {
            'operation': 'measurement',
            'length': cw_len,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf',
            },
            'integration_weights': {
                'cw_integW_cos': 'cw_integW_cos',
                'cw_integW_sin': 'cw_integW_sin',
            },
            'digital_marker': 'ON'
        },

        ##############
        # RR1 pulses #
        ##############

        'readout_pulse_in': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'ro_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW_cos': 'integW_cos',
                'integW_sin': 'integW_sin',
            },
            'digital_marker': 'ON'
        },

        'long_readout_pulse_in': {
            'operation': 'measurement',
            'length': long_ro_len,
            'waveforms': {
                'I': 'long_ro_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'long_integW_cos': 'long_integW_cos',
                'long_integW_sin': 'long_integW_sin',
            },
            'digital_marker': 'ON'
        },


        ##############
        # RR2 pulses #
        ##############

        # 'readout_pulse_Q2': {
        #     'operation': 'measurement',
        #     'length': readout_len,
        #     'waveforms': {
        #         'I': 'ro_wf_Q2',
        #         'Q': 'zero_wf'
        #     },
        #     'integration_weights': {
        #         'integW_cos': 'integW_cos',
        #         'integW_sin': 'integW_sin',
        #     },
        #     'digital_marker': 'ON'
        # },

        # 'long_readout_pulse_Q2': {
        #     'operation': 'measurement',
        #     'length': long_ro_len,
        #     'waveforms': {
        #         'I': 'long_ro_wf_Q2',
        #         'Q': 'zero_wf'
        #     },
        #     'integration_weights': {
        #         'long_integW_cos': 'long_integW_cos',
        #         'long_integW_sin': 'long_integW_sin',
        #     },
        #     'digital_marker': 'ON'
        # },


        ################
        # Q1_xy pulses #
        ################

        'saturation_pulse_Q1': {
            'operation': 'control',
            'length': sat_Q1_len,
            'waveforms': {
                'I': 'saturation_wf_Q1',
                'Q': 'zero_wf',
            },
        },

        'pi_pulse_Q1': {
            'operation': 'control',
            'length': Q1_pi_len,
            'waveforms': {
                'I': 'pi_wf_Q1',
                'Q': 'zero_wf',
            },
        },

        'pi_2_pulse_Q1': {
            'operation': 'control',
            'length': Q1_pi2_len,
            'waveforms': {
                'I': 'pi_2_wf_Q1',
                'Q': 'zero_wf',
            },
        },


        ################
        # Q2_xy pulses #
        ################

        # 'saturation_pulse_Q2': {
        #     'operation': 'control',
        #     'length': sat_Q2_len,
        #     'waveforms': {
        #         'I': 'saturation_wf_Q2',
        #         'Q': 'zero_wf',
        #     },
        # },

        # 'pi_pulse_Q2': {
        #     'operation': 'control',
        #     'length': Q2_pi_len,
        #     'waveforms': {
        #         'I': 'pi_wf_Q2',
        #         'Q': 'zero_wf',
        #     },
        # },

        # 'pi_2_pulse_Q2': {
        #     'operation': 'control',
        #     'length': Q2_pi2_len,
        #     'waveforms': {
        #         'I': 'pi_2_wf_Q2',
        #         'Q': 'zero_wf',
        #     },
        # },


        ###############
        # Q1_z pulses #
        ###############

        ###############
        # Q2_z pulses #
        ###############

        #######################
        # coupler_diff pulses #
        #######################

        ######################
        # coupler_sum pulses #
        ######################
    },

    'waveforms': {

        ####################
        # common waveforms #
        ####################

        'gauss_wf': {
            'type': 'arbitrary',
            'samples': (0.24 * gaussian(gauss_len, 0.15 * gauss_len)).tolist()
        },

        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },

        'const_wf': {
            'type': 'constant',
            'sample': 0.2
        },


        #################
        # RR1 waveforms #
        #################

        'ro_wf': {
            'type': 'constant',
            'sample': 0.22
        },

        'long_ro_wf': {
            'type': 'constant',
            'sample': 0.11
        },


        #################
        # RR2 waveforms #
        #################

        # 'ro_wf_Q2': {
        #     'type': 'constant',
        #     'sample': 0.22
        # },

        # 'long_ro_wf_Q2': {
        #     'type': 'constant',
        #     'sample': 0.11
        # },


        ###################
        # Q1_xy waveforms #
        ###################

        'saturation_wf_Q1': {
            'type': 'constant',
            'sample': 0.22
        },

        'pi_wf_Q1': {
            'type': 'arbitrary',
            'samples': (0.24 * gaussian(Q1_pi_len,  0.15 * Q1_pi_len)).tolist(), 
        },

        'pi_2_wf_Q1': {
            'type': 'arbitrary',
            'samples': (0.24 * gaussian(Q1_pi2_len, 0.15 * Q1_pi2_len)).tolist(),
        },


        #################
        # Q2 waveforms #
        #################

        # 'saturation_wf_Q2': {
        #     'type': 'constant',
        #     'sample': 0.22
        # },

        # 'pi_wf_Q2': {
        #     'type': 'arbitrary',
        #     'samples': (0.3 * gaussian(Q2_pi_len, 0.25 * Q2_pi_len)).tolist()
        # },

        # 'pi_2_wf_Q2': {
        #     'type': 'arbitrary',
        #     'samples': (0.15 * gaussian(Q2_pi_len, 0.25 * Q2_pi_len)).tolist(),
        # },

        ##################
        # Q1_z waveforms #
        ##################

        ##################
        # Q2_z waveforms #
        ##################

        ##########################
        # coupler_diff waveforms #
        ##########################

        #########################
        # coupler_sum waveforms #
        #########################

    },

    'digital_waveforms': {

        'ON': {
            'samples': [(1, 0)]
        },
    },

    'integration_weights': {

        'cw_integW_cos': {
            'cosine': [1.0] * int(cw_len / 4),
            'sine': [0.0] * int(cw_len / 4),
        },

        'cw_integW_sin': {
            'cosine': [0.0] * int(cw_len / 4),
            'sine': [1.0] * int(cw_len / 4),
        },

        'long_integW_cos': {
            'cosine': [1.0] * int(long_ro_len / 4),
            'sine': [0.0] * int(long_ro_len / 4),
        },

        'long_integW_sin': {
            'cosine': [0.0] * int(long_ro_len / 4),
            'sine': [1.0] * int(long_ro_len / 4),
        },

        'integW_cos': {
            'cosine': [1.0] * int(readout_len/4),   # np.exp(-t_d*(2*np.pi)/(2000/4)).tolist()
            'sine': [0.0] * int(readout_len/4),
        },

        'integW_sin': {
            'cosine': [0.0] * int(readout_len/4),
            'sine': [1.0] * int(readout_len/4), # np.exp(-t_d*(2*np.pi)/(2000/4)).tolist()
        },

    },

    'mixers': {

        # 'mixer_diff': [
        #     {'intermediate_frequency': diff_IF, 'lo_frequency': diff_lo, 'correction': [1, 0, 0, 1]}
        # ],

        # 'mixer_sum': [
        #     {'intermediate_frequency': sum_IF, 'lo_frequency': sum_lo, 'correction': [1, 0, 0, 1]}
        # ],s

        'mixer_RO': [
            {'intermediate_frequency': RR_1_IF, 'lo_frequency': RO_lo, 'correction': R1_corr_mat},
            #{'intermediate_frequency': RR_2_IF, 'lo_frequency': RO_lo, 'correction': R2_corr_mat},
        ],
        
        'mixer_Q1': [
            {'intermediate_frequency': Q1_IF, 'lo_frequency': Q1_lo, 'correction': Q1_corr_mat},
        ]

        # 'mixer_XY': [
        #     {'intermediate_frequency': Q1_XY_IF, 'lo_frequency': XY1_lo, 'correction': Q1_corr_mat},
        #     {'intermediate_frequency': Q2_XY_IF, 'lo_frequency': XY2_lo, 'correction': Q2_corr_mat}
        # ],

    }
}