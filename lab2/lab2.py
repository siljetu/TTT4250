import numpy as np
import matplotlib.pyplot as plt

file1 = "Free_Field_45cm_Height_d1m.etx"
file2 = "Free_Field_50cm_Height_d1m.etx"
filename_imp12 = "Imp_tube_12.etx"
filename_imp21 = "Imp_tube_21.etx"


def read_csv(filename):
    return np.genfromtxt(filename, comments='*', skip_footer=1, skip_header=1)

def plot_time_freq(time_axis, data1, data2, filename, title):
    plt.suptitle(title)
    plt.subplot(1,2,1)
    plt.plot(time_axis, data1, color="blue",label="Upper microphone")
    plt.plot(time_axis, data2, color="red",label="Lower microphone")
    plt.grid()
    plt.ylabel("Amplitude [Pa]")
    plt.xlabel("Time [ms]")
    plt.legend()

    plt.subplot(1,2,2)
    sp = np.pad(data1, (0, 512**2 - len(data1)),"constant")
    sp2 = np.pad(data2, (0, 512**2 - len(data2)),"constant")
    sp = np.fft.fft(data1, 512)
    sp = np.trim_zeros(sp, trim="fb")
    sp2 = np.fft.fft(data2,512)
    sp2 = np.trim_zeros(sp2, trim="fb")

    freq = np.fft.fftfreq(n=len(sp), d=1/fs)

    plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(sp)), color="blue", label="Upper microphone")
    plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(sp2)), color="red", label="Upper microphone")
    print('freq_shift',np.fft.fftshift(freq).shape)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":")
    plt.xscale("log")
    plt.xlim(82,2000)
    plt.ylim(15, 50)
    plt.legend()
    plt.savefig(filename)

    plt.show()

def miki_model(w0,c0,f, sigma):
    k = (w0/c0)*(1+7.81*((10**3)*f/sigma)**(-0.618)-1j*11.41*((10**3)*f/sigma)**(-0.618))
    absorp = (-1j)*(1+5.50*(((10**3)*f/sigma)**(-0.632))-1j*8.43*(((10**3)*f/sigma)**(-0.632)))*np.arctan(k*np.exp)
    return k, absorp

def calc_refl(p1,p2):
    c = 343.2*2*np.sqrt(291/293)
    s = 0.08
    x = 0.18
    sp = np.pad(p1, (0, 512 ** 2 - len(p1)), "constant")
    sp = np.trim_zeros(sp, trim="fb")
    freq = np.fft.fftfreq(n=len(sp), d=1 / fs)
    freq = np.fft.fftshift(freq)
    H_12 = p1/p2
    H_L = np.exp(-1j*((2*np.pi*freq)/c)*s)
    H_R = np.exp(1j*((2*np.pi*freq)/c)*s)
    R = ((H_12 - H_L)/(H_R - H_12))*np.exp(2*1j*2*np.pi*x)
    return R, freq

def calc_absorp(R):
    return 1-(abs(R)**2)

def calc_imp_ratio(R):
    return (1+R)/(1-R)
#####################
# Free Field
#####################
file_data1 = np.transpose(read_csv(file1))
file_data2 = np.transpose(read_csv(file2))

time_length = file_data1.shape[1]
fs = 44100
time_axis = np.linspace(0, 150, int(fs * 0.15)-1)
print('time_axis', time_axis.shape)

data1 = file_data1[1][1:int(fs * 0.15)]
data2 = file_data2[1][1:int(fs * 0.15)]


# plot_time_freq(time_axis,data1,data2, filename = "plot_freefield.pdf", title = "Free field 45 cm and 50 cm")

#################
# Impedance tube
#################

file_imp12 = np.transpose(read_csv(filename_imp12))
file_imp21 = np.transpose(read_csv(filename_imp21))


data_imp12_1 = file_imp12[1][1:int(fs * 0.15)]
data_imp12_2 = file_imp12[2][1:int(fs * 0.15)]


data_imp21_1 =file_imp21[1][1:int(fs * 0.15)]
data_imp21_2=file_imp21[2][1:int(fs * 0.15)]

# plot_time_freq(time_axis,data_imp12_1,data_imp12_2, filename="plot_imp12.pdf", title="Impulse response and FFT for impedance tube (12)")
# plot_time_freq(time_axis,data_imp21_1,data_imp21_2, filename = "plot_imp21.pdf", title = "Impulse response and FFT for impedance tube (21)")


####################################################
# Calculate impedance Z and absorption coefficient
####################################################

# Calculate Reflection Factor R for impedance tube
R, freq_axis = calc_refl(data_imp12_1,data_imp12_2)
print('R',R.shape)
print('freq_axis', freq_axis.shape)
# freq_axis = np.linspace(20,20000,6614)

# for i in range(len(R)):
#     if abs(R[i])>1:
#         True
#         # print(i)


plt.plot(freq_axis,R)
plt.show()
print("R", R)
print("abs R", abs(R))
absorp = calc_absorp(R)
plt.plot(freq_axis,absorp)
plt.show()
print("abs", absorp)


# plt.plot(time_axis,(absorp-R))
# plt.show()
###############################################################
# Compare graphically predicted absorption factor and results
###############################################################

