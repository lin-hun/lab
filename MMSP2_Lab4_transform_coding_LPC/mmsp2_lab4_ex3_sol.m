%% MMSP2 - Lab 4
%  Exercise 4 - Vocoder with voiced/unvoiced classification

clear
close all
clc

%% 1) Load the files 'a.wav' and 'shh.wav' and build a single signal
% concatenating them

[s_a,Fs] = audioread('a.wav');
s_shh = audioread('shh.wav');

s = [s_a;s_shh];

%% 2) Loop over every frame and compute voicing detection parameters
% (i.e., zcr and ste) for each one of them

% generate Hamming window
% hint: use hamming()
frame_dur = 0.04;
frame_stride = 0.01;

frame_len = round(frame_dur * Fs);
frame_step = round(frame_stride*Fs);

win = hamming(frame_len);

N = floor((length(s)-frame_len)/frame_step)+1;

% container for classification parameters
parameter = zeros(N, 2);

for n=1:N % for each frame
    
    % select a frame and multiply it with the window
    frame = s((n-1)*frame_step+1:(n-1)*frame_step+frame_len).*win;
   
    % Zero-crossing rate
    frame_zcr = sum(abs(diff(frame>0)))/frame_len;
    
    % Short-time energy
    frame_ste = sum(frame.^2);
    
    parameter(n,:) = [frame_zcr,frame_ste];

end

%% 3) Voiced / Unvoiced classification
% Compute thresholds using median()

th = median(parameter);

% Decision
voiced = parameter(:,1) < th(1) & parameter(:,2) > th(2);
% voiced(i) = 1 if the i-th frame is voiced, 0 if unvoiced


% plot the parameters with the thresholds
figure();
subplot(3,1,1);
plot(parameter(:,1),'r');
hold on;
plot(th(1)*ones(N,1),'r--');
title('ZCR');
grid on;

subplot(3,1,2);
plot(parameter(:,2),'b');
hold on;
plot(th(2)*ones(N,1),'b--');
title('STE');
grid on;

subplot(3,1,3);
plot(voiced);
title('Voiced');
grid on;

%% 4) LPC analysis and synthesis
p = 12; % order of the predictor

Fmin = 60;
Fmax = 500;
    
lagmin = floor(Fs/Fmax);
lagmax = ceil(Fs/Fmin);

% container for the synthesized speech
s_synth = zeros(length(s),1);
for n = 1:N
    
    % select a windowed frame 
    frame_idxs = (n-1)*frame_step+1:(n-1)*frame_step+frame_len;
    frame = s(frame_idxs).*win;
    
    
    %% 4a) Compute LP coefficients and prediction error
   
    [r,rlags] = xcorr(frame,frame,'normalized');
    rpos = r(rlags >=0);
    
    R = toeplitz(rpos(1:p));
    a = R\rpos(2:p+1);
    
    if n == 1
        [e, Sf_e] = filter([1; - a], 1, frame);
    else
        [e, Sf_e] = filter([1;- a], 1, frame, Sf_e);
    end
    
    
    %% 4b-1) Voiced segment:
    if voiced(n) == 1
        % Pitch detection
        rc = rpos(lagmin+1:lagmax+1);

        [~,rc_maxi] = max(rc);
        pitch_lag = lagmin+rc_maxi-1;

        % Generate impulse train
        u = zeros(length(frame),1);
        u(1:pitch_lag:end) = 1;
        
        
    %% 4b-2) Unvoiced segment:
    else
        % Generate random noise instead of impulse train
        %hint: use randn()
        u = randn(length(frame),1);
        
    end
    
    %% 4c) Normalize the energy of the excitation signal
    % (i.e., make it equal to that of the prediction error)
    u = u / std(u) * std(e);

    %% 5) Shaping filter (i.e., synthesize the frame)
    %hint: pay attention to the overlap
    if n == 1
        [frame_synth, Sf_x] = filter(1, [1; -a], u);
    else
        [frame_synth, Sf_x] = filter(1, [1; -a], u, Sf_x);
    end
    
    s_synth(frame_idxs) = s_synth(frame_idxs) + frame_synth;

end


%% 6) Listen and compare the original and synthetic signals

% normalize in [-1;1] range just for listening
s_synth = (s_synth - min(s_synth))/(max(s_synth) - min(s_synth)) *2 -1;

sound(s,Fs);
pause(5);
sound(s_synth,Fs);
