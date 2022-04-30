import matplotlib.pyplot as plt
import numpy as np

#octave_band_i = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]#,28,29,30]
octave_band = [125,250,500,1000,2000,4000, 8000, 16000]
A_weighting = [-16.1, -8.6, -3.2, 0, 1.2, 1, -1.1, -4]
x_ticks_octaveband = ["125", "250", "500", "1k", "2k", "4k", "8k", "16k"]


def plått(array, title, name_of_file, limit, use_A_Weight=True, short=False, ) -> object:
    fig, ax = plt.subplots()
    for i in range(array.shape[0]):
        if not short:
            ax.semilogx(octave_band, array[i] + (A_weighting * np.array(use_A_Weight).astype(int)),
                        label=f"Mic pos {i + 1}")
        else:
            ax.semilogx(octave_band[:-1], array[i] + (A_weighting * np.array(use_A_Weight).astype(int))[:-1],
                        label=f"Mic pos {i + 1}")
    if limit != 0:
        ax.hlines(y = limit, xmin=125, xmax = 16000, linestyle='dashed', color = 'black')
    # plot magic
    ax.hlines(y=11.8, xmin=125, xmax=16000, linestyle='dashed', color='red')
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

    #plt.legend(loc="lower right")
    plt.savefig(name_of_file)
    plt.show()

def db_to_pressure(measurements):
    return  10 ** (measurements / 10) #fjerna p0, se om d funker da


def pressure_to_db(measurements):
    return 10 * np.log10(measurements) #fjerna deling av measuremnts på p0


LW = [81.8, 77.4, 77.6, 76.2, 78.4]

def get_octave_bands(filename):
    data = np.genfromtxt(filename, delimiter=';', skip_header=1, skip_footer=3)
    selected_data = data[10:, [2,4,5,7]]
    data_press = db_to_pressure(selected_data)
    print(" sorted pressure data", data_press.shape)

    d1 = data_press.shape[1]
    d3 = 3
    d2 = data_press.shape[0]//d3
    assert (data_press.shape[0]/3).is_integer() # Safety first
    octave_band_pressure = data_press.transpose().reshape(d1, d2, d3).sum(axis=2).transpose()
    octave_bands = pressure_to_db(octave_band_pressure)
    return octave_bands

#Measurement box
meas_length = 0.465 + 0.28*2
meas_width = 0.3 + 0.28*2
meas_height = 0.25 + 0.28
meas_surface = 2 * (meas_width * meas_height) + (meas_width * meas_length) + 2 * (meas_length * meas_height)
print('meas surface', meas_surface)

octave_band_top = get_octave_bands('I_Top.csv')
octave_band_front = get_octave_bands('I_Front.csv')
octave_band_back = get_octave_bands('I_Back.csv')
octave_band_left = get_octave_bands('I_Left.csv')
octave_band_right = get_octave_bands('I_Right.csv')

combined = np.swapaxes(np.dstack((octave_band_top,octave_band_front,octave_band_back, octave_band_left,octave_band_right)),0,1).transpose()
combined_pressure = db_to_pressure(combined)


SPL_pressure = np.average(combined_pressure[:,:,0],axis = 0)
SPL_db = pressure_to_db(SPL_pressure)
print('soundpowerlevel',SPL_db)

L_W_tot = pressure_to_db(np.sum((SPL_pressure)))+A_weighting[-1]
print('LWTOT', L_W_tot)

L_I_pressure = np.average(combined_pressure[:,:,1], axis =0)
L_I_db = pressure_to_db(L_I_pressure)


LP_pressure = np.average(combined_pressure[:,:,2], axis = 0)
LP_db = pressure_to_db(LP_pressure)
print('LPdb', LP_db)

#FPL directly from data
FPI_pressure = np.average(combined_pressure[:,:,3], axis = 0)
print('fpi_data', combined_pressure[:,:,3])
print('pfi_db', combined[:,:,3])
print('PFI_pressure', FPI_pressure)
FPI_db = pressure_to_db(FPI_pressure)
print('FPI', FPI_db)

plått(np.array([FPI_db]), title='F_pI', name_of_file='fpi_met3.pdf', limit = 8.8)

#Checking that the dynamic capacity index is greater than FPI
L_d = 18.8 - 10
diff_ld_fpi = FPI_db-L_d
print('diff', diff_ld_fpi)

plått(np.array([SPL_db]), title='Sound Power Level', name_of_file='power_met3.pdf')

LI_avg = pressure_to_db(np.average(db_to_pressure(octave_band_back[:,1])+db_to_pressure(octave_band_right[:,1])+db_to_pressure(octave_band_left[:,1])+db_to_pressure(octave_band_front[:,1])+db_to_pressure(octave_band_top[:,1])))
L_W = LI_avg + 10*np.log10(meas_surface)
print('LW', L_W)

#print(L_W-L_W_db)

#erlend fikk 85.6

#FPL calculated as shown in lab compendium
# F_PL = LP_db - L_W + 10*np.log(meas_surface)


FPL_test = LP_db - L_I_db
for i in range(len(FPL_test)):
    if (FPL_test[i]<L_d):
        print('Level of accuracy fulfilled')
    else:
        print('Level of accuracy unsatisfactory')
