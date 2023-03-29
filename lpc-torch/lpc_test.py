from lpctorch import LPCCoefficients
import torchaudio
import numpy as np
from lpcfunctions import *
# Parameters
#     * sr            : sample rate of the signal ( 16 kHz )
#     * frame_duration: duration of the window in seconds ( 16 ms )
#     * frame_overlap : frame overlapping factor
#     * K             : number of linear predictive coding coefficients
sr             = 16000
frame_duration = .016
frame_overlap  = .5
K              = 32

# Initialize the module given all the parameters
lpc_prep       = LPCCoefficients(
    sr,
    frame_duration,
    frame_overlap,
    order = ( K - 1 )
)

# Get the coefficients given a signal
# torch.Tensor of size ( Batch, Samples )
X, sample_rate = torchaudio.load('./61-70968-0000.wav')
lpc_coefficients = lpc_prep( X )
