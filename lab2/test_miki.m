%% Mikis model

f = [100:1:2000];
omega = 2*pi*f;

rho_0 = 1.225;      
c_0   = 342.2;      
P_0   = 1.0132e+05; 
sigma = 9100;      
h     = 0.1;
X = f/sigma;

Z = rho_0*c_0*( 1 + 5.50*((f/sigma)*1000).^(-0.632)- i*8.43*((f/sigma)*1000).^(-0.632) ); 

k = omega/c_0 .* (-i) .* ( 11.41*((f/sigma)*1000).^(-0.618)+ i* (1 + 7.81*((f/sigma)*1000).^(-0.618) ) );

Z = -1i.*Z./tan(k*h);

R = (Z-rho_0*c_0)./(Z+rho_0*c_0);
a = 1-abs(R).^2;

figure(10)
subplot(1,2,2)
semilogx(f, real(Z))
hold on
semilogx(f, imag(Z))
semilogx(f, abs(Z))
grid on
title('Specific Impedance Z_C')
xlabel('Frequency [Hz]' )
ylabel("Magnitude")
legend('Re[Z]', "Im[Z]", "|Z|")
hold off

figure(10)
subplot(1,2,1)
semilogx(f, abs(R))
hold on
semilogx(f, abs(a))
title('Absorption-and Reflection Coefficient')
xlabel('Frequency [Hz]')
ylabel("Magnitude")
legend('Absorption', "Reflection", "Location", "best")
grid on
hold off
