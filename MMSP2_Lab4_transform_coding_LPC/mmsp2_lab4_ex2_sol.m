clc, clearvars, close all


%% 1) Load the file 'voiced_a.wav' and consider only a 300ms frame.
% Plot the magnitude of the frequency response of the frame

[s,Fs] = audioread('voiced_a.wav');

dur = 0.3;
s = s(1:dur*Fs,:);

% s = sin(2*pi*((1:dur*Fs)-1)*101/Fs);  % just a check that a pure tone exhibits an ordered set of peaks at L, 2L, 3L, etc.

N = size(s,1);
S = fft(s);

% faxes = (0:N-1)/N*Fs;
faxes = 0:Fs/N:Fs-Fs/N;

spectrum_fig = figure();
semilogx(faxes,db(abs(S)),'DisplayName','S');


xlabel('f [Hz]');
ylabel('Magnitude [dB]');
xlim([0,Fs]);
grid on;
legend();

%% 2) Perform pitch detection using auto-correlation method.
%  Consider only frequencies between 60 Hz and 500 Hz


% compute autocorrelation (hint: use two output arguments of xcorr() and 
% pay attention to MATLAB normalization)
[r,rlags] = xcorr(s,s,'coeff');

% consider correlation only for positive lags, including 0
rpos = r(rlags >=0);

r_fig = figure();
plot(rlags(rlags>=0),rpos);
title('r');
xlabel('lag');
grid on;

% find maximum peak within the accepted range
Fmin = 60;
Fmax = 500;
lagmin = floor(Fs/Fmax);
lagmax = ceil(Fs/Fmin);

rc = rpos(lagmin+1:lagmax+1);
%+1 needed to index coherently with matlab

[rc_maxv,rc_maxi] = max(rc);
%rc_maxi is indexed in matlab way (starting from 1)

r_maxi = lagmin+rc_maxi-1;
%rc_maxi-1 compensates for Matlab indexing, so that rc_maxi is 0 based indexed

r_maxv = rc_maxv;

hold on;
stem(r_maxi,r_maxv,'r');

pitch_lag = r_maxi;
pitch = Fs/pitch_lag;

fprintf('Pitch lag: %d samples - freq: %.2f Hz\n',pitch_lag,pitch);

%% 3) Compute LPC coefficients of order 12
% hint: if using correlation, consider only positive lag values.
% Use toeplitz() to build correlation matrix from correlation vector

p = 12;
R = toeplitz(rpos(1:p)); %coefficients from 0 to p-1
a = R\rpos(2:p+1); %rpos(2:p+1) : coefficients from 1 to p

% Alternatively, you can use the lpc() function, be aware of what is returned
%a_lpc = lpc(s,p);

lpc_fig = figure();
stem(a); 
title('a');
xlabel('idx');
ylabel('Value');
grid on;

A = fft([1 ;-a],N);

figure(spectrum_fig);
hold on;
plot(faxes,db(abs(1./A)),'DisplayName','1/A', 'linewidth', 2);

legend();

%% 4) Plot the prediction error and its magnitude spectrum

e = filter([1; -a],1,s);

% time_fig = figure();
% plot((0:length(e)-1)/Fs,e,'DisplayName','e');
% xlabel('Time [s]');
% ylabel('Value');
% grid on;

E = fft(e);
figure(spectrum_fig);
hold on;
plot(faxes,db(abs(E)),'DisplayName','E');

legend();

%% 5) Build an impulse train with the estimated pitch period
% hint: initialize a vector of zeros and put a 1 every 1/pitch seconds

u = zeros(length(s),1);
u(1:pitch_lag:end) = 1;

% normalize the energy: force the energy of the
% impulse train to be equal to that of the residual
u = u / std(u) * std(e); % energy normalization

% figure(time_fig);
% hold on;
% plot((0:length(u)-1)/Fs,u,'DisplayName','u');
% legend();

%% 6) Consider the impulse train as excitation and build synthetic speech
%hint: if curious use fvtool() to visualize the filter you are using.
% Can you see the formants?

s_synth = filter(1,[1; -a],u);
s_synth = s_synth/max(abs(s_synth));

S_synth = fft(s_synth,N);
figure(spectrum_fig);
hold on;
plot(faxes,db(abs(S_synth)),'DisplayName','S synth');

legend();

%% 7) Listen to the original and the synthetic speech

sound(s,Fs);
pause(1);
sound(s_synth,Fs);
