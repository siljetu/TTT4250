%% Pseudo code for lab 5 - Acoustic Measurement Terchnique
 
% Author: Robin Andre Rørstadbotnen, 27.04.2020
% Modified, 23.02.2022
% DO NOT DISTRIBUTE.
 
%% PART TWO FREE-FIELD

% All equation are taken from lecutre notes, some input equations taken
% elsewhere. Note that Figure 6 in the method document should be improved
% as the heights should be more intuitive.

%% Input
close all
clear 
clc
rho_0 = 1.186;
p_0 = 101.325;

T = 20;                              
rho = 1.225;                           
c = 343.2*sqrt( ( T + 271 ) / 293 );   


dd  = 1;                              % Horizontal distance mic - loudpeaker
hh  = 0.39;                           % Height loudspeaker
hm1 = 0.45;                           % Height mic 1
hm2 = 0.5;                            % Height mic 2


%Direct dist. travelled to mics
rd1 = sqrt ( (hh - hm1)^2 + dd^2 );  
rd2 = sqrt ( (hh - hm2)^2 + dd^2 );
rd = sqrt((hh-(hm1+hm2)/2)^2 + dd^2);
%Reflected dist. travelled to mics
rr1 = sqrt ( (hh + hm1)^2 + dd^2);   
rr2 = sqrt ( (hh + hm2)^2 + dd^2); 
rr  = sqrt ( (hh + ((hm1+hm2)/2))^2 + dd^2);
%Angle of reflection
theta1 = (pi/2 - acos(1/rr1));                              
theta2 = (pi/2 - acos(1/rr2));
theta  = (pi/2 - acos(1/rr));

%% Importing measurement
tmax      = 0.20;                        
FFfile1   = 'Free_Field_45cm_Height_d1m.etx';                
FFfile2   = 'Free_Field_50cm_Height_d1m.etx';                
Psamp1_in = importdata([FFfile1],'\t',22);
Psamp2_in = importdata([FFfile2],'\t',22); 

tt        = Psamp1_in.data(:,1) ;
dt        = tt(2) - tt(1);       
fs        = 1/dt;                 

idx_tmax  = 8e-3/(tt(2)-tt(1));               

% Extract signal of interest
p1 = Psamp1_in.data(1:idx_tmax,2);   
p2 = Psamp2_in.data(1:idx_tmax,2);
tt = tt(1:idx_tmax);


%% Making the frequency axis
n  = 2^nextpow2( size(p1,1) );  
paddingnumber = n - size(p1,1);
p1 = padarray(p1, paddingnumber, 0, "post");
p2 = padarray(p2, paddingnumber, 0, "post");

ff = fs*(0:(n-1))/n;
frecvec1 = fft(p1,n);
frecvec2 = fft(p2,n);


%% Computation

% Transfer function - eq.25 (remember component wise mulp)

H12_Free = transpose(sqrt(( p2./p1).^2));
ww = 2*pi*ff;
k =  ww / c;          

R_num = ((exp(-1i*k*rd2))/rd2)   -  H12_Free .*((exp(-1i*k*rd1))/rd1);
R_den = H12_Free.*((exp(-1i*k*rr1))/rr1)   - ((exp(-1i*k*rr2))/rr2);

R = R_num ./ R_den;
%% Impedance
scalar = (-1*rho*c) / cos(theta);
Z = (R+1) ./ (R-1);
Z = scalar * Z;
 
%% Absorption
alpha = 1 - abs(R).^2;

%% Plot
figure(32)

semilogx(ff, alpha)
hold on
semilogx(ff, abs(R))
xlim([100 2000])
ylim([0 1])
grid on
title("Absorption-and Reflection coefficient")
xlabel("Frequency [Hz]")
ylabel("Magnitude")
legend("Absorption", "Reflection")


figure(32)
semilogx(ff,real(Z),'r')
hold on
semilogx(ff,imag(Z),'b');
semilogx(ff,abs(Z),'k');
grid on
xlim([100 2000]);
title('Acoustic Impedance Z')
xlabel('Frequency [Hz]' )
ylabel("Magnitude")
legend('Re[Z]', "Im[Z]", "|Z|")
hold off



