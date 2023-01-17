clearvars
close all
clc

%% 1) Load the image 'lena512color.tiff'

im = imread('lena512color.tiff');

% we convert the image values to float
im = double(im);

%% 2) Let x be the red channel and y the green channel of the image.
%% Quantize y with PCM and DPCM (R=1,2,...,8 bits) using:
%%      2.1) y_hat(n) = a*x(n)+b
%%      2.2) y_hat(n) = randn*x(n) + randn*100

x = im(:, :, 1);
y = im(:, :, 2);

% we vectorize the signals
x = x(:);
y = y(:);

% Compute coefficients a and b with LS
% (same procedure we did in Lab1)

coeff = [x ones(size(x, 1), 1)] \ y;

a = coeff(1);
b = coeff(2);

% DPCM predictor

% please notice that for the sake of the exercise we are assuming the
% decoder knows x, a and b. In practice this is not a good idea since the 
% codec depends on the signal. A more realistic approach would have been
% encoding x_tilde = q(x) with PCM and use x_tilde for the prediction.

y_hat = a * x + b;

% Dummy DPCM predictor

rng(21);

a_d = randn;
b_d = randn * 100;

y_hat_d = a_d * x + b_d;


R = 1:8;

MSE_dpcm = zeros(length(R), 1);
MSE_dpcm_d = zeros(length(R), 1);
MSE_pcm = zeros(length(R), 1);

for ii = 1:length(R)

    % DPCM
    d = y - y_hat;
    delta_dpcm = (max(d) - min(d)) / 2^R(ii);
    d_tilde = delta_dpcm * floor(d/delta_dpcm) + delta_dpcm / 2;
    y_tilde_dpcm = d_tilde + y_hat;

    % PCM
    delta_pcm = (max(y) - min(y)) / 2^R(ii);
    y_tilde_pcm = delta_pcm * floor(y/delta_pcm) + delta_pcm/2;

    % Dummy DPCM
    d = y - y_hat_d;
    delta_dpcm = (max(d) - min(d)) / 2^R(ii);
    d_tilde = delta_dpcm * floor(d/delta_dpcm) + delta_dpcm / 2;
    y_tilde_dpcm_d = d_tilde + y_hat_d;

    % MSE
    MSE_dpcm(ii) = mean((y - y_tilde_dpcm).^2);
    MSE_pcm(ii) = mean((y - y_tilde_pcm).^2);
    MSE_dpcm_d(ii) = mean((y - y_tilde_dpcm_d).^2);

end

% SNR

SNR_dpcm = pow2db(var(y) ./MSE_dpcm);
SNR_pcm = pow2db(var(y) ./MSE_pcm);
SNR_dpcm_d = pow2db(var(y) ./MSE_dpcm_d);


%% Compare the R-D curves

figure()
plot(R, [SNR_dpcm, SNR_pcm, SNR_dpcm_d], 'linewidth', 2);
legend('DPCM', 'PCM', 'Dummy DPCM')
grid on;
xlabel('Rate [bit/symbol]');
ylabel('SNR [dB]');
set(gca, 'fontsize', 18);









