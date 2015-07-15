"SSE algorithm with Savitzky-Golay smoothing in the case of four experimental spectra"

import numpy as np
import matplotlib.pyplot as plt


def savitzky_golay(y, window_size, order, deriv=0):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')
    

"Loading experimental data"

a = plt.imread('pass_a.png')
b = plt.imread('pass_b.png')
c = plt.imread('pass_c.png')
d = plt.imread('pass_d.png')


"Conversion spectral diapason from nm to cm-1"
n = np.arange(774)
Rshift = (1.0/637.0 - (1.0/(669.0 + n*(695.5-669.0)/774.0)))*10e6


"Data binning"
R01 = 0
R02 = 0
R03 = 0
R04 = 0
for i in range (200,580):
    R01 = R01 + a[i]
    R02 = R02 + b[i]
    R03 = R03 + c[i]
    R04 = R04 + d[i]


R1 = R01[0:774:1]
R2 = R02[0:774:1]
R3 = R03[0:774:1]
R4 = R04[0:774:1]
np.savetxt("R1.txt",R1)
np.savetxt("R2.txt",R2)
np.savetxt("R3.txt",R3)
np.savetxt("R4.txt",R4)


#print R1.shape, R2.shape, R3.shape, R4.shape


"The initial estimate of fluorescence and true Raman spectra"
average = (1.0/4.0)*(R1+R2+R3+R4)
sigma = np.power((1.0/3.0)*(np.power((R1-average),2) + np.power((R2-average),2)\
+ np.power((R3-average),2) + np.power((R4-average),2)),0.5)
Sr = sigma
Sf = np.array(np.minimum(np.minimum(R1,R2),np.minimum(R3,R4)))


"Excitation shift in pixels"
shift = 5


"Starting approximation"
R2r = np.roll(R2,-1*shift)
R2r[773 - 1*shift + 1:774:1] = 0
R3r = np.roll(R3,-2*shift)
R3r[773 - 2*shift + 1:774:1] = 0
R4r = np.roll(R4,-3*shift)
R4r[773 - 3*shift + 1:774:1] = 0


for i in range (0,10000):
    Sf1 = np.roll(Sf,-1*shift)
    Sf2 = np.roll(Sf,-2*shift)
    Sf3 = np.roll(Sf,-3*shift)
    Sr1 = np.roll(Sr,-1*shift)
    Sr2 = np.roll(Sr,-2*shift)
    Sr3 = np.roll(Sr,-3*shift)
    Sr4 = np.roll(Sr,-4*shift)
    Sr6 = np.roll(Sr,-6*shift)     
    Sf1[773 - 1*shift + 1:774:1] = 0.0001
    Sf2[773 - 2*shift + 1:774:1] = 0.0001
    Sf3[773 - 3*shift + 1:774:1] = 0.0001
    Sr1[773 - 1*shift + 1:774:1] = 0.0001
    Sr2[773 - 2*shift + 1:774:1] = 0.0001
    Sr3[773 - 3*shift + 1:774:1] = 0.0001
    Sr4[773 - 4*shift + 1:774:1] = 0.0001
    Sr6[773 - 6*shift + 1:774:1] = 0.0001
    wf = R1/(Sf+Sr)+R2/(Sf+Sr1)+R3/(Sf+Sr2)+R4/(Sf+Sr3)
    wr = R1/(Sf+Sr)+R2r/(Sf1+Sr2)+R3r/(Sf2+Sr4)+R4r/(Sf3+Sr6)
    Sf = Sf*wf
    Sr = Sr*wr
 

"Savitzky-Golay smoothing"   
Sr = savitzky_golay(Sr[0:774:1,0], window_size=11, order=2)


"Ploating"
#plt.plot(Rshift,R1)
#plt.plot(R2)
#plt.plot(R3)  
plt.plot(Rshift,Sr)
plt.show()
