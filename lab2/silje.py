import numpy as np
import matplotlib.pyplot as plt

file1 = "Free_Field_45cm_Height_d1m.etx"
file2 = "Free_Field_50cm_Height_d1m.etx"
filename_imp12 = "Imp_tube_12.etx"
filename_imp21 = "Imp_tube_21.etx"
z_file = "Z.txt"


def read_csv(filename):
    return np.genfromtxt(filename, comments='*', skip_footer=1, skip_header=1, dtype= np.longdouble)


def read_csv_simple(filename):
    return np.genfromtxt(filename)


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

    plt.plot(np.fft.fftshift(freq), 20*np.log10(np.fft.fftshift(np.abs(sp))), color="blue", label="Upper microphone")
    plt.plot(np.fft.fftshift(freq), 20*np.log10(np.fft.fftshift(np.abs(sp2))), color="red", label="Upper microphone")
    print('freq_shift',np.fft.fftshift(freq).shape)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":")
    plt.xscale("log")
    plt.xlim(100,2000)
    plt.ylim(10, 50)
    plt.legend()
    plt.savefig(filename)

    plt.show()

def miki_model(w0,c0,f, sigma):
    k = (w0/c0)*(1+7.81*((10**3)*f/sigma)**(-0.618)-1j*11.41*((10**3)*f/sigma)**(-0.618))
    print(type(k))
    z = []
    for i in range(f.shape[0]):
        z.append((-1j)*(1+5.50*(((10**3)*f[i]/sigma)**(-0.632))-1j*8.43*(((10**3)*f[i]/sigma)**(-0.632)))*np.tan(k*np.exp))
    z = rho * c0* np.array(z)
    return k, z

def _nextpow2(i):
    """
    Finner neste 2'er potens av lengden i signalet ditt. Brukes før zero padding
    :param i: lengde på signalet
    :return: returner neste verdi i 2 potens
    """
    n = 1
    while n < i: n *= 2
    return n

##############################
# Plotting functions
##############################

def plot_r_alpha(x,r,a,title,filename):
    plt.plot(x,np.abs(r))
    plt.plot(x,a)
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":")
    plt.xscale("log")
    plt.xlim(100, 2000)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(filename)
    plt.show()

def impedance_plot(x,y1, y2, y3,filename):
    plt.plot(x,np.abs(y1), color = "blue")
    plt.plot(x,y2, color = "red")
    plt.plot(x,y3, color = "yellow")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":")
    plt.xscale("log")
    plt.xlim(100, 2000)
    plt.ylim(-4000, 6000)
    # plt.legend()
    plt.savefig(filename)
    plt.show()

###############################
# Impedance tube functions
###############################

def calc_refl(p1,p2):
    p1 = p1 #[:352*4].astype(np.clongdouble)
    p2 = p2 #[:352*4].astype(np.clongdouble)

    c = 343
    s = 0.08
    x = 0.18
    ######  Regner ut FFT for de to signalene for å bruke dem i TF utregningen

    sp = np.pad(p1, (0, _nextpow2(len(p1)) - len(p1)), "constant")
    sp2 = np.pad(p2, (0, _nextpow2(len(p2)) - len(p2)), "constant")

    sp = np.trim_zeros(sp, trim="fb")
    sp2 = np.trim_zeros(sp2, trim="fb")

    sp = np.fft.fftshift(np.fft.fft(sp, _nextpow2(len(p1))))
    sp2 = np.fft.fftshift(np.fft.fft(sp2, _nextpow2(len(p2))))

    freq = np.fft.fftshift(np.fft.fftfreq(n=len(sp), d=1 / fs))

    ###### Regner ut TF og R #############
    omega =  2 * np.pi * freq
    H_L = []
    H_R = []
    for i in omega:
        H_R.append(np.exp(s*1j*i / c))
        H_L.append(np.exp(s*-1j*i / c))
    H_L = np.array(H_L)
    H_R = np.array(H_R)

    H_12 = sp2/sp   ## Transfer function blir regnet ut ifra trykk i frekvensdomenet.

    #H_L = np.exp(-1j*(omega/c)*s)
    #H_R = np.exp(1j*(omega/c)*s)
    temp = []
    for i in omega:
        temp.append(np.exp((2j*2*np.pi*i*x) / c))
    temp = np.array(temp)
    R = ((H_12 - H_L)/(H_R - H_12))*temp
    return R, freq

def calc_absorp(R):
    return 1 - (np.abs(R))**2

def calc_imp_ratio(R):
    return ((1 + R) * (rho * c)) / (1 - R)

##############################
# Free-field functions
##############################
def refl_freefield(p1,p2):
    sp = np.pad(p1, (0, _nextpow2(len(p1)) - len(p1)), "constant")
    sp2 = np.pad(p2, (0, _nextpow2(len(p2)) - len(p2)), "constant")

    sp = np.trim_zeros(sp, trim="fb")
    sp2 = np.trim_zeros(sp2, trim="fb")

    sp = np.fft.fftshift(np.fft.fft(sp, _nextpow2(len(p1))))
    sp2 = np.fft.fftshift(np.fft.fft(sp2, _nextpow2(len(p2))))
    freq = np.fft.fftshift(np.fft.fftfreq(n=len(sp), d=1 / fs))

    # H_12 = sp2/sp

    H_12 = np.sqrt(sp2*sp) #du er mistenkt for å gjøre at det blir feil mann

    ww = 2 * np.pi * freq
    k = ww / c

    R_num =((np.exp(-1j*k*rd2))/rd2)-H_12*(np.exp(-1j*k*rd1))/rd1
    R_den = H_12 * (np.exp(-1j*k*rr1))/rr1 - ((np.exp(-1j*k*rr2))/rr2)
    R = R_num/R_den
    return R, freq


#####################
# Constants
#####################
rho = 1.225
T = 20
c = 343.2*np.sqrt((T+271)/293)


dd = 1  #Horizontal distance between microphone and loudspeaker
hh = 0.39 # Loudspeaker height
hm1 = 0.45 # Microphone 1 height
hm2 = 0.5 # Microphone 2 height

# Tot. direct dist. traveled to mic
rd1 = np.sqrt((hh-hm1)**2+dd**2)
rd2 = np.sqrt((hh-hm2)**2+dd**2)
rd = np.sqrt((hh-(hm1+hm2)/2)+dd**2)
# Tot. refl. dist. travelled to mic 1
rr1 = np.sqrt((hh+hm1)**2+dd**2)
rr2 = np.sqrt((hh+hm2)**2+dd**2)
rr = np.sqrt((hh+((hm1+hm2)/2))**2+dd**2)
# Reflection angle
theta1 = (np.pi/2 - np.arccos(1/rr1))
theta2 = (np.pi/2 - np.arccos(1/rr2))
theta = (np.pi/2 - np.arccos(1/rr))



#####################
# Free Field
#####################
file_data1 = np.transpose(read_csv(file1))
file_data2 = np.transpose(read_csv(file2))

time_length = file_data1.shape[1]
fs = 44100
time_axis = np.linspace(0, 150, int(fs * 0.15)-1)
# print('time_axis', time_axis.shape)

data1 = file_data1[1][1:int(fs * 0.15)]
data2 = file_data2[1][1:int(fs * 0.15)]


plot_time_freq(time_axis,data1,data2, filename = "plot_freefield.pdf", title = "Free field 45 cm and 50 cm")

#################
# Impedance tube
#################

file_imp12 = np.transpose(read_csv(filename_imp12))
file_imp21 = np.transpose(read_csv(filename_imp21))


data_imp12_1 = file_imp12[1][1:int(fs * 0.15)]
data_imp12_2 = file_imp12[2][1:int(fs * 0.15)]


data_imp21_1 =file_imp21[1][1:int(fs * 0.15)]
data_imp21_2=file_imp21[2][1:int(fs * 0.15)]

plot_time_freq(time_axis,data_imp12_1,data_imp12_2, filename="plot_imp12.pdf", title="Impulse response and FFT for impedance tube (12)")
plot_time_freq(time_axis,data_imp21_1,data_imp21_2, filename = "plot_imp21.pdf", title = "Impulse response and FFT for impedance tube (21)")


####################################################
# Calculate impedance Z and absorption coefficient
####################################################

# Calculate Reflection Factor R for impedance tube
R, freq_axis = calc_refl(data_imp12_1,data_imp12_2)
R.tofile("reflection_imptube.txt", sep = ",")
absorp = calc_absorp(R)
# print("R", R.shape)
print("1-R", 1-R)

plot_r_alpha(freq_axis, R, absorp, title = "Reflection and absorption",filename="r_abs_imptube.pdf")

# imp = calc_imp_ratio(R)
# print("imp",imp.shape)

# impedance_plot(freq_axis, imp, imp.real, imp.imag, filename="imp_ratio_freefield.pdf")
Z = read_csv_simple(z_file)
Z = np.array(Z)
print("freq_axis", freq_axis.shape)
print("Z", Z.shape)
impedance_plot(freq_axis, Z, Z.real, Z.imag, filename = "imp_from_matlab.pdf")
# plot_r_alpha(freq_axis, R_test, calc_absorp(R_test), filename="testest-r.pdf")
impedance_plot(freq_axis,calc_imp_ratio(R), calc_imp_ratio(R).real,calc_imp_ratio(R).imag, filename = "tes_imp.pdf")

##########################
# Free-field
##########################

# One-shot calibration

R_free, freq_free = refl_freefield(data1,data2)
absorp_free = calc_absorp(R_free)
# plot_r_alpha(freq_free,R_free,absorp_free, filename="r_alpha_free.pdf")

#############################
# Testing for 1-R and 1+R
#############################

# test_axis = np.linspace(20,20000,int(8192/2))
# len = R.shape[0]/2
# plt.plot(test_axis,1-R[int(R.shape[0]/2):], color="red")
# plt.plot(test_axis,1+R[int(R.shape[0]/2):], color="blue")
# plt.xscale("log")
# plt.xlim(100,20000)
# plt.ylim(0,3)
# plt.show()
#
# plot_r_alpha(freq_axis,(1-R),(1+R),filename="test.pdf")
# print("size vec", data_imp21_2.shape)
# ###############################################################
# # Compare graphically predicted absorption factor and results
# ###############################################################
# freqs = np.linspace(100,2000,2000)
# sigma = 9100
# miki = miki_model(2*np.pi*freqs,c,freqs, sigma)
