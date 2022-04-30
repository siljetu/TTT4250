import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

filename = "method1.csv"
filename2 = "method2.csv"



def read_csv(filename, usecols):  # np arange creates array from 1 to 18, skipping the first row
    return np.genfromtxt(filename, comments='*', delimiter=';', skip_header=1, usecols=usecols, dtype=float)

def rad_to_angle(arg):
    return arg * 180 / np.pi

def degree_to_rad(arg):
    return arg* np.pi /180

def integrand(data, angle):
    return (data ** 2) * np.abs(np.sin(angle))

def DI(arg):
    return 10*np.log10(arg)

angle1 = np.array([90,75,60,45,30,15])
angle2 = np.array([15, 30, 45, 60, 75, 90])
angle_axis = np.linspace(-90, 90, 13)

##############################
# Calculating D
##############################


row1 = [3.77, 3.64, 3.25, 2.88, 3.16, 3.36, 3.19, 3.47, 3.74, 3.78, 3.31, 3.4, 3.3]
row2 = [7.52, 6.04, 5.32, 4.58, 5.04, 6.38, 6.6, 6.34, 5.29, 5.03, 5.66, 5.88, 5.46]
row3 = [3.83, 3.8, 3.56, 2.45, 2.555, 5.02, 5.48, 3.7, 1.855, 2.825, 3.83, 4.06, 4.06]

row1_max = np.max(row1)
row2_max = np.max(row2)
row3_max = np.max(row3)

row1_scaled = row1/row1_max
row2_scaled = row2/row2_max
row3_scaled = row3/row3_max


# integral_test = integrate.quad(integrand(np.flip(row1_scaled[:6]), degree_to_rad(angle1)),0, (np.pi/2))

# integral_test = integrate.simps(integrand(np.flip(row1_scaled[:6]), degree_to_rad(angle1)),angle_axis[:6])
# print("integraltest", integral_test)
# print("Dtest", 2/integral_test )
# print("DITESTT SIMPSON", DI(2/integral_test))


# nptrapz = np.trapz(integrand(np.flip(row1_scaled[:6]), degree_to_rad(angle1)), np.flip(angle_axis[:6]))
# print("npTrapz", nptrapz)
# print("Dtest npTrapz", 2/nptrapz )
# print("DI test npTrapz", DI(2/nptrapz))

print("row1", row1_scaled)
print("row test", row1_scaled[:6])
print("row test", row1_scaled[7:])
print("angle2", degree_to_rad(angle2).shape)
print("angleaxis", angle_axis[6:])
# print("test", degree_to_rad(90))
# print("testing degree to ang",degree_to_rad(angle1))
# print("testing reversed row1_scaled", np.flip(row1_scaled))
D_8kHz_1quad = integrate.simps(2 / integrand(row1_scaled[:6], degree_to_rad(angle1)),degree_to_rad(angle1))
D_12kHz_1quad = integrate.simps(2 / integrand(row2_scaled[:6], degree_to_rad(angle1)), degree_to_rad(angle1))
D_16kHz_1quad = integrate.simps(2 / integrand(row3_scaled[:6], degree_to_rad(angle1)), degree_to_rad(angle1))
print("D_8_1 degree", rad_to_angle(D_8kHz_1quad))
print("D_12_1 degree", rad_to_angle(D_12kHz_1quad))
print("D_16_1 degree", rad_to_angle(D_16kHz_1quad))

# print("integrand test 1", integrand(row1_scaled[7:], degree_to_rad(angle2)))
#
# print("integrand test2 ", integrand(row2_scaled[7:], degree_to_rad(angle2)))
#
# print("integrand test3", integrand(row2_scaled[7:], degree_to_rad(angle2)))

D_8kHz_2quad = integrate.simps(2 / integrand(row1_scaled[7:], degree_to_rad(angle2)),degree_to_rad(angle2))
D_12kHz_2quad = integrate.simps(2 / integrand(row2_scaled[7:], degree_to_rad(angle2)),degree_to_rad(angle2))
D_16kHz_2quad = integrate.simps(2 / integrand(row3_scaled[7:], degree_to_rad(angle2)),degree_to_rad(angle2))
print("D-8_2 degree", rad_to_angle(D_8kHz_2quad))
print("D-12_2 degree", rad_to_angle(D_12kHz_2quad))
print("d_16_2 degree", rad_to_angle(D_16kHz_2quad))

print("D_8_1 rad", D_8kHz_1quad)
print("D_12_1 rad", D_12kHz_1quad)
print("D_16_1 rad", D_16kHz_1quad)
print("D-8_2 rad", D_8kHz_2quad)
print("D-12_2 rad", D_12kHz_2quad)
print("d_16_2 rad", D_16kHz_2quad)

############################
# Calculating DI
############################

DI_8kHz_1quad = DI(np.abs(D_8kHz_1quad))
DI_12kHz_1quad = DI(np.abs(D_12kHz_1quad))
DI_16kHz_1quad = DI(np.abs(D_16kHz_1quad))

DI_8kHz_2quad = DI(D_8kHz_2quad)
DI_12kHz_2quad = DI(D_12kHz_2quad)
DI_16kHz_2quad = DI(D_16kHz_2quad)

print("DI_8Khz_1quad", DI_8kHz_1quad)
print("DI_12Khz_1quad", DI_12kHz_1quad)
print("DI_16Khz_1quad", DI_16kHz_1quad)

print("DI_8Khz_2quad", DI_8kHz_2quad)
print("DI_12Khz_2quad", DI_12kHz_2quad)
print("DI_16Khz_2quad", DI_16kHz_2quad)