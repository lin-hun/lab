clc
clearvars
close all

%% Generate 1000 samples with gaussian distribution and variance=3
rng(21)

x_var = 3;
x = randn(1000,1) * sqrt(x_var);

%% Quantize with a scalar mid-rise quantizer with fixed quantization step delta=2

delta = 2