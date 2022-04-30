import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

octave_band = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
A_weighting = np.array([-16.1, -8.6, -3.2, 0, 1.2, 1, -1.1, -4])
x_ticks_octaveband = ["125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
p0 = 20 * 10 ** (-6)
LW_RSS_A_tot = 90.5
r = np.array([1.524, 2.480, 2.899, 3.469, 2.220])
LW_RSS_short = np.array([79.8, 81.0, 80.9, 84.9, 85.1, 82.7, 79.2])
A_weighting_modded = A_weighting[:-1]
mic_pos = 5

MEASUREMENT = {
    "LFEQ": 0,
    "LFMAX": 1,
    "LFMIN": 2,
    "LFE": 3
}

unmodded_file_ref = "Trykk_Lab1.csv"


names_for_plot = ['Mic pos 1', 'Mic pos 2', 'Mic pos 3', 'Mic pos 4', 'Mic pos 5', 'Background noise']


def read_csv(filename):
    return np.genfromtxt(filename, skip_header=1, delimiter=';')


# Array of type:  [ROWS, OCTAVEBANDS]
def plått(array, title,name_of_file, limit, use_A_Weight=True, short=False) -> object:# limit,
    fig, ax = plt.subplots()
    for i in range(array.shape[0]):
        if not short:
            ax.semilogx(octave_band, array[i] + (A_weighting * np.array(use_A_Weight).astype(int)),
                        label=f"Mic pos {i + 1}")
        else:
            ax.semilogx(octave_band[:-1], array[i] + (A_weighting * np.array(use_A_Weight).astype(int))[:-1],
                        label=f"Mic pos  {i + 1}")
    # for i in range(array.shape[0]):
    #     if not short:
    #         ax.semilogx(octave_band, array[i] + (A_weighting * np.array(use_A_Weight).astype(int)),
    #                     label=names_for_plot[i])
    #     else:
    #         ax.semilogx(octave_band[:-1], array[i] + (A_weighting * np.array(use_A_Weight).astype(int))[:-1],
    #                     label=names_for_plot[i])
    if limit != 0:
        ax.hlines(y = limit, xmin=125, xmax = 16000, linestyle='dashed', color = 'black')
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

    # plt.legend(loc="lower right")
    plt.savefig(name_of_file)
    plt.show()


def background_noise_correction(L_marked_pi):
    L_pi = 10 * np.log10(10 ** (L_marked_pi / 10) - 10 ** (L_marked_pi / 10))


def db_to_pressure(measurements):
    return 10 ** (measurements /10)


def pressure_to_db(measurements):
    return 10 * np.log10(measurements)




# Reading the files
data = read_csv(unmodded_file_ref)[:-1, 4:]  # All rows minus the last, header is fucked anyway, fuck the first 4 colums
print("data shape", data.shape)

bg_noise = data[[0, 5, 6, 9, 12], :]
print("bg noise shape", bg_noise.shape)

ref_noise = data[[1, 4, 8, 11, 14], :]
print("ref noise shape", ref_noise.shape)

test_noise = data[[2, 3, 7, 10, 13], :]
print("test", test_noise.shape)

method_2_data = data[[15, 16, 17, 18, 19]]
print('method 2 data', method_2_data.shape)


# Trash the first 15 columns as they are not relevant for the octavebands
# Split into microphone positions, measurement and octavebands
reference_splitted = ref_noise[:, 15:].reshape(-1, 4, 11)[:, :,3:]  # The three first octave bands are invalid due to measurement equipment limitation
background_splitted = bg_noise[:, 15:].reshape(-1, 4, 11)[:, :, 3:]
test_splitted = test_noise[:, 15:].reshape(-1, 4, 11)[:, :, 3:]
method_2_splitted = method_2_data[:, 15:].reshape(-1, 4, 11)[:, :, 3:]


print("Microphone positions, measurements, octavebands", reference_splitted.shape)
to_plot_ref = reference_splitted[:, MEASUREMENT["LFEQ"], :]
to_plot_back = background_splitted[:, MEASUREMENT["LFEQ"], :]
print('toplotback', to_plot_back)
to_plot_test = test_splitted[:, MEASUREMENT["LFEQ"], :]
to_plot_method2 = method_2_splitted[:, MEASUREMENT["LFEQ"], :]
print('toplotmet2',to_plot_method2.shape)
#######################################################################
# Method 1
#######################################################################
#Average A-weighted SPL
a_to_plot_method1 = to_plot_test + A_weighting
meas_pressure_met1 = db_to_pressure(a_to_plot_method1)
pres_avg_method1 = np.average(meas_pressure_met1, axis = 0)
avg_spl_metod1 = pressure_to_db(pres_avg_method1)



# Calculate average background noise per octaveband
background_pressure = db_to_pressure(to_plot_back)
print("PRessure_vec shape", background_pressure.shape)

pres_avg = np.average(background_pressure, axis=0)

# From pressure to dB
average_background = pressure_to_db(pres_avg)

#################################


# plått([to_plot_ref, to_plot_test], "Reference sound source", name_of_file='ref_method1_plt.pdf')
# plått(to_plot_test, 'Test sound source', name_of_file='test_method1_plt.pdf')

# Calculating delta Lpi,RSS and delta Lpi,TS
delta_ref = to_plot_ref - average_background
delta_test = to_plot_test - average_background

# plått(delta_ref, 'Delta Lpi RSS', name_of_file='deltapi_ref_method1_plt.pdf', limit=15)
# plått(delta_test, 'Delta Lpi TS', name_of_file='deltapi_test_method1_plt.pdf', limit =15)

# Calculating delta Lf
print('plot ref', to_plot_ref.shape)
print('lw_rss', LW_RSS_short.shape)

delta_lf_short_rss = to_plot_ref[:, :-1] - LW_RSS_short + 11 + 20 * np.log10(r.reshape(1, -1).transpose())
delta_lf_short_ts = to_plot_test[:, :-1] - LW_RSS_short + 11 + 20 * np.log10(r.reshape(1, -1).transpose())

# plått(delta_lf_short_rss, 'Delta LF_RSS', name_of_file='deltalf_rss_method1_plt.pdf', limit = 7 , short=True)
# plått(delta_lf_short_ts, 'Delta LF_TS', name_of_file='deltalf_ts_method1_plt.pdf', limit = 7 , short=True)

# Sound Pressure Levels
print('LPI_TS', to_plot_test)
print('toplotref', to_plot_ref)

# Calculation of Sound Power Levels
sum_sound_pressure_short = db_to_pressure(to_plot_test[:, :-1] - to_plot_ref[:, :-1])
diff_test_ref = to_plot_test[:,:-1]-to_plot_ref[:,:-1]
diff_test_ref = np.array(diff_test_ref)
diff_test_ref_pressure = db_to_pressure(diff_test_ref)
avg_diff = np.average(diff_test_ref_pressure, axis = 0)
db_avg_diff = pressure_to_db(avg_diff)
L_W1_octavebands = LW_RSS_short + db_avg_diff
L_W1_octavebands = np.array(L_W1_octavebands)
LW_A_weight = L_W1_octavebands + A_weighting[:-1]
# print('LWAWEIGHT', LW_A_weight)


LW_tot1 = 10*np.log10(np.sum(10**(0.1*LW_A_weight)))
# print('LW_a_1', LW_tot1)


sound_pressure_avg = pressure_to_db(np.average(db_to_pressure(sum_sound_pressure_short)))

L_W_short = LW_RSS_short + 10 * np.log10((1 / mic_pos) * sound_pressure_avg)

# plått(np.array([L_W_short]), title='Sound Power Levels, L_W', name_of_file='lw_met1.pdf', short=True)

######################################################
# Method 2
######################################################

# Environment Correction

# Equivalent absorption area
alpha = 0.05
Surface = 2 * (6.042 * 5.174) + 2 * (5.174 * 8.501) + 2 * (8.501 * 6.042)
print('Sv', Surface)
Absorption_area = alpha * Surface
print('abs area', Absorption_area)
meas_length = 0.465 + 2
meas_width = 0.3 + 2
meas_height = 0.25 + 1
meas_surface = 2 * (meas_width * meas_height) + (meas_width * meas_length) + 2 * (meas_length * meas_height)
print('meas surface', meas_surface)
K_2A = 10 * np.log10(1 + (4 * meas_surface / Absorption_area))
print('KA2', K_2A)
# K_2A = 0.045

# Calculating averaged A-weighted background noise

weighted_background = average_background + A_weighting
# plått(np.array([weighted_background]), title ='A-weighted background noise', name_of_file ='background_method2_plt.pdf')

# Calculating average sound pressure levels per octave band
background_pressure = db_to_pressure(to_plot_back)
pres_avg = np.average(background_pressure, axis=0)

# From pressure to dB
average_background = pressure_to_db(pres_avg)

#Average SPL
a_to_plot_method2 = to_plot_method2 + A_weighting
print('a_met2', to_plot_method2)
meas_pressure = db_to_pressure(a_to_plot_method2)
print('toplotmet2', to_plot_method2)
pres_avg_method2 = np.average(meas_pressure, axis = 0)
print('presavg', pres_avg_method2)
avg_spl_metod2 = pressure_to_db(pres_avg_method2)
print('avgsplmet2',avg_spl_metod2.shape)
print('avg_back', average_background.shape)

#Preparing to plot SPL and background together
spl_back_met2 = np.vstack((a_to_plot_method2, average_background))
print('spl_back_met2', spl_back_met2.shape)
# plått(to_plot_method2, title="Sound pressure level", name_of_file='spl_met2.pdf')
# plått(spl_back_met2, title="Sound pressure level and A-weighted background noise", name_of_file='spl_back_met2.pdf')

#Average spl for all methods
avg_spl_metod3 = [75.65487634, 78.38296303, 81.8487299,  79.05088953, 74.63620086, 71.27643254,
 64.73294099, 53.79423521] #Copypasted from the other code
avg_spl_all_met = np.vstack((avg_spl_metod2, avg_spl_metod1, avg_spl_metod3))
print('avglsplallmetods',avg_spl_all_met )
# plått(avg_spl_all_met, title='Averaged A-weighted SPL for all methods', name_of_file='avg_spl_all_met.pdf')

#averaged spl
# plått(np.array([avg_spl_metod2]), title='Averaged A-weighted SPL', name_of_file='avg_spl_met2.pdf')
#Sound pressure level
# plått(to_plot_method2, title='Sound pressure level', name_of_file='spl_met2.pdf')

# Calculate background noise correction
delta_LA = avg_spl_metod2 - average_background
print('delta la', delta_LA)
plått(np.array([delta_LA]), title='Delta LA', name_of_file='deltala_method2.pdf',limit = 10)

# Calculate A-weighted Sound Power Level
K_1A = 0
LW2_array = avg_spl_metod2 + A_weighting
LW_2 = pressure_to_db(np.sum(db_to_pressure(avg_spl_metod2 + A_weighting))) - K_1A - K_2A + 10 * np.log10(meas_surface)
print('avgaslp shape', avg_spl_metod2.shape)
print('LW_2', LW_2)
# plått(np.array([LWA_2, LW_2], dtype=object), title='Sound Power Level L_W,A', name_of_file='lwa_method2_zweight.pdf', use_A_Weight=False)
#plått(np.array([LWA_2]), title='Sound Power Level L_W,A', name_of_file='lwa_method2_aweight.pdf', )

LW3 = [80.5509246, 73.06390169, 76.44831435, 74.00392732, 70.24746077, 67.90200648,
 62.5901889,  52.15249025] #Copypasted from the other code
print('lw2', LW2_array[:-1].shape)
print('lw1', LW_RSS_short.shape)
all_LW = np.vstack((L_W_short,LW2_array[:-1], LW3[:-1]))
# plått(all_LW, title= 'Sound power levels for the three methods', name_of_file='sound_power_all_met.pdf', short=True)
