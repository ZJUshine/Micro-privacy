import numpy as np
import soundfile as sf
import scipy.linalg, scipy.interpolate
import librosa
import pandas as pd
from scipy.signal import lfilter
from tqdm import tqdm


class CELP(object):
    '''
    This class defines a generic CELP codec.

    Parameters
    ---
    wave_path: str
        The path of original audio.
    save_path: str
        The path of audio after codec.
    frame_duration: float
        The duration of each frame, in seconds. Default=0.02.
    LPCorder: int
        The LPC order of LPC analysis. Default=16.
    gamma: float
        The denominator coefficient of perceptual weighted filter. Default=0.85.
    SCB_num: int
        The number of stochastic codewords. Default=1024.
    codebook_exist: bool
        Whether the codebook already exists. If True, use the specified codebook, else randomly generate the codebook. Default=False.
    pitch: tuple
        The searching range of pitch. Default=(50, 500) with 16k sample rate.
    target_sr: int
        The sample rate of CELP codec. Default=16000. If doesn't match, resample to the target. 
    save_wav: bool
        Whether to save the audio after codec. Default=False.
    '''

    def __init__(self, wave_path, save_path, 
                    frame_duration=0.02, 
                    subframe_num=4, 
                    LPCorder=16, 
                    gamma = 0.85, 
                    SCB_num=1024, 
                    codebook_exist=False, 
                    pitch=(50, 500), 
                    target_sr=16000,
                    save_wav = True
                    ):
        # self.data, self.sr = sf.read(wave_path)
        self.wave_path = wave_path
        # self.target_sr = target_sr
        self.sr = target_sr
        self.orig_data, self.orig_sr = sf.read(self.wave_path)
        if self.orig_sr != self.sr:
            self.data = librosa.resample(y=self.orig_data, orig_sr=self.orig_sr, target_sr=self.sr)
        else: 
            self.data = self.orig_data
        self.save_path = save_path
        self.frame_duration = frame_duration  # duration of per frame
        self.subframe_num = subframe_num
        self.LPCorder = LPCorder
        self.gamma = gamma
        self.N = int(self.sr * self.frame_duration)  # frame length
        self.M = int(self.N / self.subframe_num)   # subframe length
        self.SCB_num = SCB_num   # Stochastic codebook index length

        ## load codebook
        if codebook_exist == True:
            self.SCB = pd.read_csv('/home/xsl/micprivacy/code/codebook.csv', sep=',', header=None)
            self.SCB = np.array(self.SCB)
            if self.SCB.shape[0] != self.SCB_num:
                raise ValueError('shape[0] of exist codebook does not match attribute [self.SCB_num] !')
            if self.SCB.shape[1] != self.M:
                raise ValueError('shape[1] of exist codebook does not match attribute [self.M] !')
        else:
            self.SCB = np.random.randn(self.SCB_num, self.M)   # create Stochatic codebook index length 1024*M
        
        self.pitch = pitch  # range of pitch, Hz
        self.Pidx = (int(self.sr/self.pitch[1]), int(self.sr/self.pitch[0]))
        self.Ndata = len(self.data)  # length of data
        self.frame_num = np.fix(self.Ndata / self.N).astype(np.int16)  # No. of frames
        print('File name: %s, sample rate: %d' %(self.wave_path, self.sr))
        self.new_data = np.zeros(self.Ndata)
        self.ak = np.zeros((self.frame_num, self.LPCorder + 1))
        self.save_wav = save_wav

    def run(self):
        ## initialize output signals
        # new_data = np.zeros((self.Ndata, 1))  # synthesized signal
        e = np.zeros((self.Ndata,1))  # excitation signal
        SCB_indx = np.zeros((self.frame_num, self.subframe_num))  # rows are excitation  
        theta0 = np.zeros((self.frame_num, self.subframe_num))  # parameters per frame
        P = np.zeros((self.frame_num, self.subframe_num)) 
        b = np.zeros((self.frame_num, self.subframe_num))
        ebuf = np.zeros(self.Pidx[1])  # vectors with previous excitation
        ebuf2 = ebuf.copy()
        bbuf = 0  # samples
        Zf = np.zeros(self.LPCorder)
        Zw = np.zeros(self.LPCorder)
        Zi = np.zeros(self.LPCorder)  # memory hangover in filters

        
        print('Frame number: %d' %(self.frame_num))
        for i in tqdm(range(self.frame_num)):
            n = np.arange(i*self.N, (i+1)*self.N)   # time index of current speech frame
            SCB_indxf, theta0f, akf, Pf, bf, ebuf, Zf, Zw = self.Analysis_by_synthesis(self.data[n], bbuf, ebuf, Zf, Zw, i)   # extract params with AbS

            self.new_data[n], ebuf2, Zi = self.celpsyns(akf, SCB_indxf, theta0f, Pf, bf, ebuf2, Zi)   # synthesis new frame
            # inter2 = time.time()
            
            ## store params 
            SCB_indx[i, :] = SCB_indxf
            theta0[i, :] = theta0f
            P[i, :] = Pf 
            b[i, :] = bf
            bbuf = bf[-1]  # last estimated b used in next frame
            # print('Abs time = %ss, syns time = %ss' %(inter1-start, inter2-start))
        if self.save_wav == True:
            sf.write(self.save_path, self.new_data, self.sr)
            print('Save output file: %s' %(self.save_path))
        return self.new_data


    def Analysis_by_synthesis(self, x, bbuf, ebuf, Zf, Zw, i):
        '''
        Analyzing parameters of one frame.

        Parameters
        ---
        x: ndarray
            samples of one frame
        bbuf: ndarray
            buffer of LTP filter gain coefficients
        ebuf: ndarray
            buffer of the past excitation with length of self.Pidx[1]
        Zf: ndarray
            buffer to store the state of filter 1/A(z/c)
        Zw: ndarray
            buffer to store the state of filter A(z)/A(z/c)
        i: int
            the ith frame to be analyzed

        Return
        ---
        SCB_indxf: ndarray
            codebook indexes of one frame (4 subframes)
        theta0f: ndarray
            gains correspond to excitations
        ak: ndarray
            LPC coefficients
        Pf: ndarray
            Pitch delay of one frame (4 subframes)

        '''
        SCB_indxf = np.zeros(self.subframe_num).astype(int)
        theta0f = np.zeros(self.subframe_num)
        Pf = np.zeros(self.subframe_num).astype(int)
        bf = np.zeros(self.subframe_num)
        
        if np.mean(np.power(x,2)) < 0.00001:  # judge whether muted
            ak = [1] + [0]*self.LPCorder
            ak = np.array(ak)
        else:
            ak, _ = self.lpc_coeff(x, self.LPCorder)
        self.ak[i] = ak  # store original LPC coeff. ak 

        ##---------------------------------------------------------------------------------##
        # adjust lsf to modify formants
        lsf = self.lpc_to_lsf(ak)
        # add your own transformation function here
        ak = self.lsf_to_lpc(lsf)
        ##---------------------------------------------------------------------------------##



        for j in range(self.subframe_num):
            n = np.arange(j*self.M, (j+1)*self.M)  # time index of current subframe
            SCB_indxf[j], theta0f[j], Pf[j], bf[j], ebuf, Zf, Zw = self.celpexcit(x[n], ak, self.ak[i], bbuf, ebuf, Zf, Zw)
            bbuf = bf[-1]  # last estimated b

        return SCB_indxf, theta0f, ak, Pf, bf, ebuf, Zf, Zw


    def celpexcit(self, x, ak, ak_orig, b, ebuf, Zf, Zw):
        '''
        determine the parameters of the excitation
        ------------    Gain, theta0  ----------------
        | Gaussian |            |     |      1       |
        | codebook |----------->X---->| ------------ |----> e(n)
        |   cb     | SCB_indx         | 1 - b*z^(-P) |
        ------------                  ----------------
        '''
        F = len(ebuf)   # No. of previous excitation samples
        L = self.M     # subframe length
        Pidx = self.Pidx
        cb = self.SCB
        ar = ak.copy()
        ac = ar.copy()
        ar_orig = ak_orig.copy()
        ac_orig = ar_orig.copy()
        ci = self.gamma
        for i in range(1, len(ar)):
            ac[i] *= ci
            ac_orig[i] *= ci
            ci *= self.gamma

        ## the rows of E are the signal e(n-P) for Pidx[0] < P < Pidx[1]
        E = np.zeros((Pidx[1]-Pidx[0]+1, L))

        ##  For P < L, the signal e(n-P) is estimated as the output of the pitch, 
        ## filter with zero input. b*ebuf(F-P+1:F) is memory hangover in the filter
        if Pidx[0] < L:
            if Pidx[1] < L:
                P1 = Pidx[1]
            else:
                P1 = L
            for P in range(Pidx[0], P1):  ## 求延迟P小于40的E
                i = P - Pidx[0]
                E[i], _ = lfilter([1], np.append(1, np.append(np.zeros(P-1), -1)), np.zeros(L), zi=b*ebuf[F-P:].reshape(-1))  # 相当于把ebuf中的-P到-1放到E[i,:]中
        ## For P >= L, the signal e(n-P) can be obtained from previous excitation
        ## samples only, buffered in the vector ebuf.
        if Pidx[1] >= L:
            if Pidx[0] >= L:
                P0 = Pidx[0]
            else:
                P0 = L
            row = ebuf[F-P0: F-P0+L]
            col = ebuf[F-P0:: -1]
            i = np.arange(P0 - Pidx[0], Pidx[1]-Pidx[0]+1)
            E[i] = scipy.linalg.toeplitz(col, row)  # 利用toeplitz矩阵性质创建延迟P大于等于40的 E
        
        ## First, b and P are determined to minimize the error energy between
        ## X(z)*W(z) and b*E(z)*z^(-P) / A(z/c).
        zeta_w0, Zw = lfilter(ar, ac, x, zi=Zw);      # zeta_w0 = X(z)*W(z).
        # zeta_w0, Zw = lfilter(ar_orig, ac, x, zi=Zw);      # zeta_w0 = X(z)*W(z).
        Zeta_w2, _ = lfilter([1], ac, E, zi=np.tile(Zf, (E.shape[0],1)));            # Zeta_w2 = E(z)*z^(-P) / A(z/c)
                                                # for Pidx(1) < P < Pidx(2).
        P_w2  = np.sum(np.multiply(Zeta_w2, Zeta_w2), axis=1)         # Vector with signal power for each P. (160-16+1),

        P_w02 = np.matmul(Zeta_w2, zeta_w0)        # Vector with cross-correlations for each P. (160-16+1),40 * 40,1 = (160-16+1),

        temp = np.divide(np.multiply(P_w02,P_w02), P_w2+0.1**6)   # Find index Phat of max value. avoid 0/0 by adding a small number
        Phat = temp.argmax()
        if Phat > self.Pidx[1]-self.Pidx[0]: Phat=self.Pidx[1]-self.Pidx[0]
        P = Phat + Pidx[0]         # Offset index with first P.
        b = np.abs(P_w02[Phat]/(P_w2[Phat]+0.1**6))  # b must be larger than 0,
        if b > 1.4:                             # and less than 1.4.
            b = 1.4
        
        ## Find the signal e(n-P) based on the estimated b and P.
        if P < L:
            e, _ = lfilter([1], np.append([1], np.append(np.zeros((1, P-1)), -b)), np.zeros(L), zi=b*ebuf[F-P:])
        else:
            e = b * ebuf[F-P:F-P+L]
        
        ## Now, k and theta0 are determined to minimize the error energy between
        ## X(z)*W(z) - b*E(z)*z^(-P)/A(z/c) and theta0*rho_k(z)/A(z/c).
        
        zeta_w0 = zeta_w0 - lfilter([1], ac, e, zi=Zf)[0]  # Subtract b*E(z)*z^(-P)/A(z/c).  40,1
        Zeta_w1 = lfilter([1], ac, cb)             # Zeta_w1 = rho_k(z) / A(z/c)  1024,40
                                                # for all index k in codebook.
        P_w1  = np.sum(np.multiply(Zeta_w1, Zeta_w1), axis=1)         # Vector with signal power for each P. 1024*1
        P_w01 = Zeta_w1 @ zeta_w0;        # Vector with cross-correlations for each P.  1024,1

        temp = np.divide(np.multiply(P_w01, P_w01), P_w1)
        k = temp.argmax()             # Find index k of max value,
        theta0 = P_w01[k]/P_w1[k]     # and gain theta0 using this k.

        ## Find the signal e(n) based on the estimated b, P, theta0, and k.
        if (P < L):
            e, _ = lfilter([1], np.append(1, np.append(np.zeros(P-1), -b)), theta0*cb[k], zi=b*ebuf[F-P:])
        else:
            e = theta0*cb[k] + b*ebuf[F-P: F-P+L]
        ebuf = np.append(ebuf[L:F], e)               # Update e(n) buffer.

        _, Zf = lfilter([1], ac, e, zi=Zf)          # Update memory hangover in 1/A(z/c).

        return k, theta0, P, b, ebuf, Zf, Zw


    def celpsyns(self, ak, k, theta0, P, b, ebuf, Zi):
        '''
        ------------  Gain, theta0 ----------------      ---------
        | Gaussian |          |    |      1       |      |   1   |
        | codebook |--------->X--->| ------------ |----->| ----- |--->
        |   cb     | rho_k(n)      | 1 - b*z^(-P) | e(n) |  A(z) | x(n)
        ------------               ----------------      ---------
        '''
        F = len(ebuf)  # No. of previous excitation samples
        e = np.zeros(self.N)
        L = self.M  # block length
        cb = self.SCB
        for j in range(self.subframe_num):
            n = np.arange(j*L, (j+1)*L)

            ## find the signal e(n) based on the parameters b, P, theta0 and k
            if P[j] < L:
                Zp = b[j]*ebuf[F-P[j]: F]
                e[n], _ = lfilter([1], np.append([1], np.append(np.zeros(P[j]-1), -b[j])), theta0[j]*cb[k[j]], zi=Zp)
            else:
                e[n] = theta0[j]*cb[k[j]] + b[j]*ebuf[F-P[j]: F-P[j]+L]
            ebuf = np.append(ebuf[L:F], e[n])

        x, Zi = lfilter([1], ak, e, zi=Zi)
        return x, ebuf, Zi

    def lpc_coeff(self, s, p):
        """
        Use autocorrelation method to calculate the coefficient a_k of LPC model. Use Levinson-Durbin algorithm to solve Yule-Walker equations.
        :param p: the order of LPC model
        :return ak: (NDarray) LPC ploe coefficient  G: gain
        """
        n = len(s)
        ## calculate autocorrelation function
        Rp = np.zeros(p)
        for i in range(p):
            Rp[i] = np.sum(np.multiply(s[i + 1:n], s[:n - i - 1]))
        Rp0 = np.matmul(s, s.T)
        Ep = np.zeros((p, 1))
        k = np.zeros((p, 1))
        a = np.zeros((p, p))
        ## Levinson-Durbin algorithm
        Ep0 = Rp0
        k[0] = Rp[0] / Rp0
        a[0, 0] = k[0]
        Ep[0] = (1 - k[0] * k[0]) * Ep0
        ## recursion from i=1
        if p > 1:
            for i in range(1, p):
                k[i] = (Rp[i] - np.sum(np.multiply(a[:i, i - 1], Rp[i - 1::-1]))) / Ep[i - 1]
                a[i, i] = k[i]
                Ep[i] = (1 - k[i] * k[i]) * Ep[i - 1]
                for j in range(i - 1, -1, -1):
                    a[j, i] = a[j, i - 1] - k[i] * a[i - j - 1, i - 1]
        ak = np.zeros(p + 1)
        ak[0] = 1
        ak[1:] = -a[:, p - 1]  # A(z)=1+\sum_{k=1}^p a_k z^{-k}=\sum_{i=0}^p a_k z^{-k}
        G = np.sqrt(Ep[p - 1])
        return ak, G
    
    def lpc_to_lsf(self, all_lpc):
        if len(all_lpc.shape) < 2:
            all_lpc = all_lpc[None]
        order = all_lpc.shape[1] - 1
        all_lsf = np.zeros((len(all_lpc), order))
        for i in range(len(all_lpc)):
            lpc = all_lpc[i]
            lpc1 = np.append(lpc, 0)
            lpc2 = lpc1[::-1]
            sum_filt = lpc1 + lpc2
            diff_filt = lpc1 - lpc2

            if order % 2 != 0:
                deconv_diff, _ = scipy.signal.deconvolve(diff_filt, [1, 0, -1])
                deconv_sum = sum_filt
            else:
                deconv_diff, _ = scipy.signal.deconvolve(diff_filt, [1, -1])
                deconv_sum, _ = scipy.signal.deconvolve(sum_filt, [1, 1])

            roots_diff = np.roots(deconv_diff)
            roots_sum = np.roots(deconv_sum)
            angle_diff = np.angle(roots_diff)
            angle_sum = np.angle(roots_sum)
            angle_diff = angle_diff[np.where(angle_diff > 0)]
            angle_sum = angle_sum[np.where(angle_sum > 0)]
            lsf = np.sort(np.hstack((angle_diff, angle_sum)))
            if len(lsf) != 0:
                all_lsf[i] = lsf
        return np.squeeze(all_lsf)


    def lsf_to_lpc(self, all_lsf):
        if len(all_lsf.shape) < 2:
            all_lsf = all_lsf[None]
        order = all_lsf.shape[1]
        all_lpc = np.zeros((len(all_lsf), order + 1))
        for i in range(len(all_lsf)):
            lsf = all_lsf[i]
            zeros = np.exp(1j * lsf)
            sum_zeros = zeros[::2]
            diff_zeros = zeros[1::2]
            sum_zeros = np.hstack((sum_zeros, np.conj(sum_zeros)))
            diff_zeros = np.hstack((diff_zeros, np.conj(diff_zeros)))
            sum_filt = np.poly(sum_zeros)
            diff_filt = np.poly(diff_zeros)

            if order % 2 != 0:
                deconv_diff = scipy.signal.convolve(diff_filt, [1, 0, -1])
                deconv_sum = sum_filt
            else:
                deconv_diff = scipy.signal.convolve(diff_filt, [1, -1])
                deconv_sum = scipy.signal.convolve(sum_filt, [1, 1])

            lpc = .5 * (deconv_sum + deconv_diff)
            # Last coefficient is 0 and not returned
            all_lpc[i] = lpc[:-1]
        return np.squeeze(all_lpc)
    
    def lpcff(self, ak, nfft):
        p1 = len(ak)
        ff = 1 / np.fft.fft(ak, nfft)
        return ff


## test example
if __name__ == '__main__':
    codec = CELP(wave_path=r'/home/lxc/datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac',
                save_path=r'61-70968-0000test.flac',
                )
    codec.run()
