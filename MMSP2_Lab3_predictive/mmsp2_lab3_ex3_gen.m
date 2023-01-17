clearvars
close all
clc

%% Load the file

[x, Fs] = audioread('gb.wav');

%% Quantize x with 5 bit

R = 5;

max_x = max(x);
min_x = min(x);
    
% Determine delta
delta = (max_x-min_x)/(2^R);

% Quantize x
x_q = delta * floor(x/delta) + delta/2;

%% Compare x and x_q
figure
subplot(211);
stft(x);
xlabel('x');
subplot(212);
stft(x_q);
xlabel('x_q');

figure
subplot(211);
histogram(x);
xlabel('x');
subplot(212);
histogram(x_q);
xlabel('x_q');

%% Save the quantized version

audiowrite('ns.wav', x_q, Fs);