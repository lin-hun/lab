%% MMSP2 - Lab 2
%  Exercise 1 - Basic scalar Quantization

clc
clearvars
close all

%% Generate 1000 samples with gaussian distribution and variance=3
rng(21);

x_var = 3;
x = randn(1000,1) * sqrt(x_var);


%% Quantize with a scalar mid-rise quantizer with fixed quantization step delta=2


delta = 2;

y1 = delta * floor(x/delta); 
% y1 = delta * floor(x/delta)+ delta/2; %this does not include 0 in the reproduction levels

figure();
plot(x,y1,'.', 'markersize', 12);
xlabel('in');
ylabel('out');
grid on;
title('Mid-rise');
set(gca, 'fontsize', 18);


%% Quantize with a scalar mid-tread quantizer with fixed quantization step delta=2

delta = 2;

y2 = delta * floor((x + delta/2)/delta); 
% y2 = delta * floor((x + delta/2)/delta)+ delta/2; %this does not include 0 in the reproduction levels


figure();
plot(x,y2,'.', 'markersize', 12);
xlabel('in');
ylabel('out');
grid on;
title('Mid-tread');
set(gca, 'fontsize', 18);


%% Quantize with a scalar mid-tread quantizer with M=4 output levels

M = 4;
delta = (max(x)-min(x))/M;

y3 = delta * floor((x + delta/2)/delta);

% be aware of the 
y3_values = unique(y3);
y3(y3 == y3_values(end)) = y3_values(end-1);

figure();
plot(x,y3,'.', 'markersize', 12);
xlabel('in');
ylabel('out');
grid on;
title('M = 4');
set(gca, 'fontsize', 18);

%% Quantize using cb = [-5,-3,-1,0,1,3,5] as reproduction levels
% and th = [-4,-2,-0.5,0.5,2,4] as thresholds

cb = [-5,-3,-1,0,1,3,5];
th = [-4,-2,-0.5,0.5,2,4];

th = [-inf, th, inf];

y4 = zeros(size(x));
for level = 1:length(cb)
    mask = x >= th(level) &  x < th(level+1);
    y4(mask) = cb(level);
end

figure();
plot(x,y4,'.', 'markersize', 12);
xlabel('in');
ylabel('out');
grid on;
title('Custom');
set(gca, 'fontsize', 18);

%% Power and var
e = y1 - x;

Px = mean(x.^2);
Pe = mean(e.^2);

sig2x = var(x);
sig2e = var(e);

snr_p = Px / Pe;
snr_s = sig2x / sig2e;


%% Compute distortion using MSE for each one of the above quantizers

mse1 = mean((y1-x).^2);
mse2 = mean((y2-x).^2);
mse3 = mean((y3-x).^2);
mse4 = mean((y4-x).^2);

%% Compute SNR for each one of the above quantizers


snr1 = var(x) / mse1;
snr2 = var(x) / mse2;
snr3 = var(x) / mse3;
snr4 = var(x) / mse4;

% if we want the snr to be in dB we can use one of the following:
snr1_db = 10*log10(snr1);
snr2_db = db(snr2,'power');
disp([num2str(snr1),',',num2str(snr1_db)])


