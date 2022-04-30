import matplotlib.pyplot as plt
import numpy as np


A0=10
T0=0.5
c = 343
rho_s = (15.33+10.08)
rho = 1.2041
S = 1.21*1.18
octave_band = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
octave_band = np.array(octave_band)
octave_band_short = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
octave_band_short = np.array(octave_band_short)
omega = octave_band*int(2*np.pi)
omega = np.array(omega)
omega_short = octave_band_short*int(2*np.pi)
print("omega short", omega_short)
omega_short = np.array(omega_short)
A_weighting = np.array([-16.1, -8.6, -3.2, 0, 1.2, 1, -1.1, -4])
x_ticks_octaveband = ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800", "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k", "4k", "5k", "6.3k", "8k", "10k", "12.5k", "16k", "20k"]
x_ticks_octaveband_short = ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800", "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k", "4k", "5k"]
A_weighting_modded = A_weighting[:-1]
mic_pos = 5

MEASUREMENT = {
    "LFEQ": 0,
    "LFMAX": 1,
    "LFMIN": 2,
    "LFE": 3
}

unmodded_file_ref = "pressure.csv"
reverb_file = "reverberation.csv"


names_for_plot = ['Mic pos 1', 'Mic pos 2', 'Mic pos 3', 'Mic pos 4', 'Mic pos 5', 'Background noise']


def read_csv(filename, skipheader, skipfooter):
    return np.genfromtxt(filename, skip_header=skipheader, skip_footer = skipfooter, delimiter=';')


# Array of type:  [ROWS, OCTAVEBANDS]
def plått(array, title,name_of_file, use_A_Weight=True, short=False) -> object:# limit,
    fig, ax = plt.subplots()
    for i in range(array.shape[0]):
        if not short:
            ax.semilogx(octave_band, array[i] + (A_weighting * np.array(use_A_Weight).astype(int)),
                        label=f"Mic pos {i + 1}")
        else:
            ax.semilogx(octave_band[:-1], array[i] + (A_weighting * np.array(use_A_Weight).astype(int))[:-1],
                        label=f"Mic pos  {i + 1}")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    if not short:
        ax.set_xticks(octave_band)
        ax.set_xticklabels(x_ticks_octaveband)
    else:
        ax.set_xticks(octave_band[:-1])
        ax.set_xticklabels(x_ticks_octaveband[:-1])
    plt.savefig(name_of_file)
    plt.show()

def plått_multi(n_m_array, title, name_of_file, lengend_array):
    assert len(n_m_array.shape) == 2  # hvis denne failer har du en funky array
    fig, ax = plt.subplots()

    for i in range(n_m_array.shape[0]):
        ax.semilogx(octave_band, n_m_array[i],
                    label=f"{lengend_array[i]}")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    plt.legend(bbox_to_anchor = (1,0.5), loc="center left")
    plt.savefig(name_of_file)
    plt.show()

def plått_leveldiff(n_m_array, title, name_of_file, lengend_array):
    assert len(n_m_array.shape) == 2  # hvis denne failer har du en funky array
    fig, ax = plt.subplots()

    for i in range(n_m_array.shape[0]):
        ax.semilogx(octave_band_short, n_m_array[i],
                    label=f"{lengend_array[i]}")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    plt.legend(bbox_to_anchor = (1,0.5), loc="center left")
    plt.savefig(name_of_file)
    plt.show()


def plått_simple(array, title,name_of_file):
    fig, ax = plt.subplots()
    ax.semilogx(octave_band, array)
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    # plt.legend(loc="lower right")
    plt.savefig(name_of_file)
    plt.show()

def plått_soundreduction(array, title,name_of_file):
    fig, ax = plt.subplots()
    ax.semilogx(octave_band_short, array)
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    # plt.legend(loc="lower right")
    plt.savefig(name_of_file)
    plt.show()

def plått_dn(n_m_array, title, name_of_file, lengend_array):
    assert len(n_m_array.shape) == 2  # hvis denne failer har du en funky array
    fig, ax = plt.subplots()

    for i in range(n_m_array.shape[0]):
        ax.semilogx(octave_band_short, n_m_array[i],
                    label=f"{lengend_array[i]}")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    plt.legend(bbox_to_anchor = (1,0.5), loc="center left")
    plt.savefig(name_of_file)
    plt.show()


def plått_theory(n_m_array, title, name_of_file, lengend_array):
    # assert len(n_m_array.shape) == 2  # hvis denne failer har du en funky array
    fig, ax = plt.subplots()

    for i in range(n_m_array.shape[0]):
        ax.semilogx(octave_band_short, n_m_array[i],
                    label=f"{lengend_array[i]}")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    plt.legend(bbox_to_anchor = (1,0.5), loc="center left")
    plt.savefig(name_of_file)
    plt.show()





def db_to_pressure(measurements):
    return 10 ** (measurements /10)


def pressure_to_db(measurements):
    return 10 * np.log10(measurements)

def R(L1,L2,S,A):
    return L1-L2 + 10 *np.log10(S/A)

def R_random(R0):
    return R0-10*np.log10(R0)

def R_field(R0):
    return R0 - 5

def R_marked(D, S, A):
    return D+ 10*np.log10(S/A)

def R0():
    return 10*np.log10(1+((omega_short *rho_s)/(2*rho*c))**2)

def D(L1,L2):
    return L1-L2

def Dn(D,A):
    return D-10*np.log10(A/A0)

def equivalent_absorption_area(V, T):
    return (0.16*V) /T

def DnT(D, T):
    return D+10*np.log10(T/T0)

def average_spl(data):
    meas_pressure = db_to_pressure(data)
    pressure_average = np.average(meas_pressure, axis = 0)
    avg_db = pressure_to_db(pressure_average)
    return avg_db

def average_spl_two_arrays(data1, data2):
    data1_pressure = db_to_pressure(data1)
    data2_pressure = db_to_pressure(data2)
    data_total_pressure = np.vstack((data1_pressure,data2_pressure))
    average_pressure = np.average(data_total_pressure, axis = 0)
    average_db_data = pressure_to_db(average_pressure)
    return average_db_data

def background_noise_correction(background, data):
    return 10*np.log10(10**((background+data)/10)-10**((background)/10))

# Reading the files
data = read_csv(unmodded_file_ref, skipheader=1, skipfooter=0)[:, 4:]  # All rows minus the last, header is fucked anyway, fuck the first 4 colums
print("data shape", data.shape)
print("data", data)

reverb_data = read_csv(reverb_file, skipheader=3, skipfooter=9)[3:,[4]] #Avgerage revberation time per 1/3-octave band
print("reverb", reverb_data)


bg_noise = data[[10, 13], :]
print("bg noise shape", bg_noise.shape)

source_lower = data[[0,3,4,7,8], :]
print("source lower shape", source_lower.shape)

source_upper = data[[1,2,5,6,9], :]
print("source upper shape", source_upper.shape)

receiver_lower = data[[12,14,17,18,21], :]
print("receiver lower shape", receiver_lower.shape)

receiver_upper = data[[11,15,16,19,20], :]
print("receiver upper shape", receiver_upper.shape)



# Trash the first 15 columns as they are not relevant for the octavebands
# Split into microphone positions, measurement and octavebands
source_upper_splitted = source_upper[:, 15:].reshape(-1, 4, 33)[:, :,9:]  # The 9 first octave bands are invalid due to measurement equipment limitation
print("source upper slitted", source_upper_splitted.shape)
source_lower_splitted = source_lower[:, 15:].reshape(-1, 4, 33)[:, :, 9:]
receiver_upper_splitted = receiver_upper[:, 15:].reshape(-1, 4, 33)[:, :, 9:]
receiver_lower_splitted = receiver_lower[:, 15:].reshape(-1, 4, 33)[:, :, 9:]
background_splitted = bg_noise[:, 15:].reshape(-1, 4, 33)[:, :, 9:]

source_upper_splitted_short = source_upper[:, 15:].reshape(-1, 4, 33)[:, :,9:-6]  # The 9 first octave bands are invalid due to measurement equipment limitation
print("source upper slitted", source_upper_splitted_short.shape)
source_lower_splitted_short = source_lower[:, 15:].reshape(-1, 4, 33)[:, :, 9:-6]
receiver_upper_splitted_short = receiver_upper[:, 15:].reshape(-1, 4, 33)[:, :, 9:-6]
receiver_lower_splitted_short = receiver_lower[:, 15:].reshape(-1, 4, 33)[:, :, 9:-6]
background_splitted_short = bg_noise[:, 15:].reshape(-1, 4, 33)[:, :, 9:-6]




to_plot_source_upper =  source_upper_splitted [:, MEASUREMENT["LFEQ"], :]
to_plot_source_lower = source_lower_splitted[:, MEASUREMENT["LFEQ"], :]
to_plot_receiver_upper= receiver_upper_splitted[:, MEASUREMENT["LFEQ"], :]
to_plot_receiver_lower = receiver_lower_splitted[:, MEASUREMENT["LFEQ"], :]
to_plot_receiver_lower = receiver_lower_splitted[:, MEASUREMENT["LFEQ"], :]
to_plot_back = background_splitted[:, MEASUREMENT["LFEQ"], :]

to_plot_source_upper_short =  source_upper_splitted_short [:, MEASUREMENT["LFEQ"], :]
to_plot_source_lower_short = source_lower_splitted_short[:, MEASUREMENT["LFEQ"], :]
to_plot_receiver_upper_short= receiver_upper_splitted_short[:, MEASUREMENT["LFEQ"], :]
to_plot_receiver_lower_short = receiver_lower_splitted_short[:, MEASUREMENT["LFEQ"], :]
to_plot_receiver_lower_short = receiver_lower_splitted_short[:, MEASUREMENT["LFEQ"], :]
to_plot_back_short = background_splitted_short[:, MEASUREMENT["LFEQ"], :]


##################
# Method 1
##################

# Averaged SPL
average_upper_source = average_spl(to_plot_source_upper)
average_lower_source = average_spl(to_plot_source_lower)
average_upper_receiver = average_spl(to_plot_receiver_upper)
average_lower_receiver = average_spl(to_plot_receiver_lower)
average_background = average_spl(to_plot_back)

average_source= average_spl_two_arrays(average_upper_source,average_lower_source)
average_receiver = average_spl_two_arrays(average_lower_receiver,average_upper_receiver)
average_source_short= average_spl_two_arrays(average_upper_source,average_lower_source)
average_receiver_short = average_spl_two_arrays(average_lower_receiver,average_upper_receiver)

averaged_vector = np.vstack((average_upper_source,average_lower_source,average_upper_receiver,average_lower_receiver,average_background))
averaged_vector_correct = np.vstack((average_source,average_receiver,average_background))


plått_multi(averaged_vector_correct,title="Average Sound Pressure Levels", name_of_file="avg_spl_all.pdf", lengend_array=["Source room", "Receiving room", "Background noise"])


average_upper_source_short = average_spl(to_plot_source_upper_short)
average_lower_source_short = average_spl(to_plot_source_lower_short)
average_upper_receiver_short = average_spl(to_plot_receiver_upper_short)
average_lower_receiver_short = average_spl(to_plot_receiver_lower_short)
average_background_short = average_spl(to_plot_back)

########################
# Background noise
########################

bg_and_receving = db_to_pressure(average_receiver)+db_to_pressure(average_background)
diference_bgnoise =db_to_pressure(average_background)-bg_and_receving


average_background[23] = background_noise_correction(average_background[23],average_receiver[23])
average_background[23] = background_noise_correction(average_background[23],average_receiver[23])
average_background[22] = background_noise_correction(average_background[22],average_receiver[22])
average_background[19] = background_noise_correction(average_background[19],average_receiver[19])

bg_and_receving_after = db_to_pressure(average_receiver)+db_to_pressure(average_background)
diference_bgnoise_after =db_to_pressure(average_background)-bg_and_receving


averaged_vector_correct = np.vstack((average_source,average_receiver,average_background))

plått_multi(averaged_vector_correct,title="Average Sound Pressure Levels", name_of_file="avg_spl_all.pdf", lengend_array=["Source room", "Receiving room", "Background noise"])


#####################################################
# Equivalent sound absorption area receiving room
#####################################################

height = 5.58
width = 4.11
length = 4.58
volume_receiving_room = height*width*length
# stigning = 30/reverb_data
# T60 = 60/stigning
# T60 = np.array(T60).squeeze()
T60 = reverb_data.squeeze()



####################################
# Calculating T60 from T30
####################################

absorp_t60 = equivalent_absorption_area(volume_receiving_room,T60)
absorp_t60 = absorp_t60.squeeze()




################################################################
# Calculating level difference between the source room and the receiving room
################################################################
D_upper = D(average_upper_source_short,average_upper_receiver_short)
D_lower = D(average_lower_source_short,average_lower_receiver_short)

D_calc = D(average_source, average_receiver)
D_calc_short = D_calc[:-6]

#########################################################
# Calculating normalized level difference Dn (receiver)
#########################################################
Dn_upper = Dn(D_upper,absorp_t60)
Dn_lower = Dn(D_lower, absorp_t60)



Dn_calc = average_spl_two_arrays(Dn_lower,Dn_upper)

DnT_upper = DnT(D_upper, T60)
DnT_lower = DnT(D_lower, T60)

DnT_calc = average_spl_two_arrays(DnT_upper,DnT_lower)


leveldiff_vec = np.vstack((D_calc_short, Dn_calc, DnT_calc))
plått_leveldiff(leveldiff_vec,title="Level difference", name_of_file="leveldiff.pdf", lengend_array=["D", "Dn", "DnT"])

#####################################################
# Sound Reduction Index
#####################################################
sound_red_index = R_marked(D_upper,S, absorp_t60)
plått_soundreduction(sound_red_index,"Sound Reduction Index R",name_of_file="sound_red_index.pdf")




###################################################
# Calculating R, R_marked, R_field, R_random
###################################################

sound_red_index_test = R_marked(D_upper,S,absorp_t60.squeeze())
plått_soundreduction(sound_red_index_test,"Sound Reduction Index","sound_red_t60.pdf")


sound_R0 = R0()
sound_red_field= R_field(sound_R0)
print("field",sound_red_field.shape)
sound_red_random = R_random(sound_R0)
print("random",sound_red_random.shape)

print("sound red index", sound_red_index.shape)
theoretical_R = np.vstack((sound_red_index,sound_red_field,sound_red_random))
plått_theory(theoretical_R,"R', R_field, R_random", "theory_reduction.pdf",["R'", "R_field", "R_random"])

