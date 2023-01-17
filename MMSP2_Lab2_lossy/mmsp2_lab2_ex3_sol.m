%% MMSP2 - Lab 2
%  Exercise 3 - Scalar quantization

clearvars
close all
clc

%% 1) Suppose that you are sampling the output of a sensor at 10 kHz for 10 seconds.
%%    Quantize the output with a uniform quantizer at 10 bit per sample.
%%    Assume that the pdf of the signal is gaussian with mean 0 V and variance 4 V^2.
%%    What is the bit rate of the quantized signal?


Fs = 10e3;
dur = 10;


n_bit = 10;
var_x = 4;
mean_x = 0;

N = Fs*dur; %number of samples
bitrate = n_bit*Fs; %bit per second = n_bit/sample * sample/second

%% 2) What would be a reasonable choice for the quantization step?

x = randn(N,1)*sqrt(var_x)+mean_x;

xmin = min(x);
xmax = max(x);

M = 2^n_bit; %we have 2^10 possible values, a reasonable choice for M is 2^10

delta = (xmax-xmin)/M;

xq = round(x/delta)*delta;

xqval = unique(xq);
xq(xq==xqval(end)) = xqval(end-1);

%% 3) What is the MSE?

% quantization error
e = xq-x;

% MSE
mse = mean(e.^2);

%% What is the resulting SNR?

snr = 10*log10(var_x/mse);