import numpy as np
import matplotlib.pyplot as plt


filename = "method1.csv"
filename2 = "method2.csv"


def read_csv(filename, usecols):  # np arange creates array from 1 to 18, skipping the first row
    return np.genfromtxt(filename, comments='*', delimiter=';', skip_header=1, usecols=usecols, dtype=float)


def rad_to_angle(arg):
    return arg * 180 / np.pi

#

##########################
# Method 1
##########################

data = read_csv(filename, usecols=np.arange(1, 18 + 1))
data_reshaped = data.reshape(data.shape[0], data.shape[1] // 2, 2)


twoer_average_data = np.average(data_reshaped, axis=2)
twoer_average_data_transp = twoer_average_data.transpose()
twoer_max = np.max(twoer_average_data_transp)
twoer_decibel = 20 * np.log10(twoer_average_data_transp / twoer_max)


freq_axis = np.linspace(8000, 16000, 9).repeat(3).reshape(9, 3)
plt.plot(freq_axis, twoer_decibel, label=["0 degrees", "45 degrees", "90 degrees"])
plt.legend(loc="lower center")
plt.grid(True)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Normalized amplitude")
plt.savefig("met1_freq_resp.pdf")
plt.show()

#############################
# Method 2
#############################

data_met2 = read_csv(filename2, usecols=np.arange(1, 26 + 1))
data_reshaped_met2 = data_met2.reshape(data_met2.shape[0], data_met2.shape[1] // 2, 2)
twoer_average_data_met2 = np.average(data_reshaped_met2, axis=2)


twoer_average_data_met2_transp = twoer_average_data_met2.transpose()
twoer_max_met2 = np.max(twoer_average_data_met2_transp)
twoer_decibel_met2 = 20 * np.log10(twoer_average_data_met2_transp / twoer_max_met2)

angle_axis = np.linspace(-90, 90, 13)
angles = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
angles = np.array(angles)
angle_rad = angles / (np.pi * 180)
x_ticks_angles = ["-90", "-75", "-60", "-45", "-30", "-15", "0", "15", "30", "45", "60", "75", "90"]


fig, ax = plt.subplots()
plt.plot(angle_axis, twoer_decibel_met2, label=["8 kHz", "12  kHz", "16 kHz"])
plt.legend(loc="lower center")
ax.set_xticks(angles)
ax.set_xticklabels(x_ticks_angles)
plt.grid(True)
plt.xlabel("Angle [°]")
plt.ylabel("Normalized amplitude")
plt.savefig("met2_freq_resp.pdf")
plt.show()
