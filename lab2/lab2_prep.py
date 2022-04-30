import numpy as np
import matplotlib.pyplot as plt

file1 = "Free_Field_45cm_Height_d1m.ext"
file2 = "Free_Field_50cm_Height_d1m.ext"


def read_csv(filename):
    return np.genfromtxt(filename)



file_data1 = np.transpose(read_csv(file1))
file_data2 = np.transpose(read_csv(file2))

time_length = file_data1.shape[1]
fs = 44100
time_axis = np.linspace(0, 8, int(fs * 0.008))

data1 = file_data1[1][:int(fs * 0.008)]
data2 = file_data2[1][:int(fs * 0.008)]

plt.subplot(1,2,1)
plt.plot(time_axis, data1, color="blue",label="Upper microphone")
plt.plot(time_axis, data2, color="red",label="Lower microphone")
plt.grid()
plt.ylabel("Amplitude [Pa]")
plt.xlabel("Time [ms]")
plt.legend()

plt.subplot(1,2,2)
sp = np.pad(data1, (0, 512 - len(data1)),"constant")
sp2 = np.pad(data2, (0, 512 - len(data2)),"constant")
sp = np.fft.fft(data1, 512)
sp = np.trim_zeros(sp, trim="fb")
sp2 = np.fft.fft(data2,512)
sp2 = np.trim_zeros(sp2, trim="fb")

freq = np.fft.fftfreq(n=len(sp), d=1/fs)

plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(sp)), color="blue", label="Upper microphone")
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(sp2)), color="red", label="Upper microphone")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.xscale("log")
plt.grid(which="major")
plt.grid(which="minor", linestyle=":")
plt.xscale("log")
plt.xlim(82,2000)
plt.ylim(15, 50)
plt.legend()

plt.show()




