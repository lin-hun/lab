%% MMSP2 - Lab 3
%  Exercise 2 - Predictive coding

clear
close all
clc

%% Load the stereo file ‘mso.wav’ and define xl and xr as
%% the left and right channels, respectively.

[x, Fs] = audioread('mso.wav');

xl = x(:,1);
xr = x(:,2);

N = size(x, 1);

%% 1) Build a DPCM codec. Use the left channel xl as a signal and
%%      1.1) xl(n-1)
%%      1.2) xr(n)
%%      1.3) dummy 5*xl(n)
%% as predictor.
%%    Use PCM to initialize the codec.

R = 1:8;

MSE_l = zeros(length(R), 1);
MSE_r = zeros(length(R), 1);
MSE_d = zeros(length(R), 1);
MSE_pcm = zeros(length(R), 1);

for ii = 1:length(R)

    xl_l_tilde = zeros(N, 1); % prediction of L using L
    xl_r_tilde = zeros(N, 1); % prediction of L using R
    xl_d_tilde = zeros(N, 1); % dummy prediction

    % first sample: PCM
    max_xl = max(xl);
    min_xl = min(xl);
    delta_pcm = (max_xl - min_xl) / 2^R(ii);

    xl_l_tilde(1) = delta_pcm * floor(xl(1)/delta_pcm) + delta_pcm/2; % quantization (midrise)

    % next samples: DPCM

    % difference beween signal and prediction of the signal
    % without the quantization step, this would be the residual
    d_l_aux = xl(2:end) - xl(1:end-1);
    delta_l = (max(d_l_aux) - min(d_l_aux)) / (2^R(ii));
    
    d_r_aux = xl - xr;
    delta_r = (max(d_r_aux) - min(d_r_aux)) / (2^R(ii));
    
    d_d_aux = xl - 5*xl;
    delta_d = (max(d_d_aux) - min(d_d_aux)) / (2^R(ii));

    for nn = 2:N
        
        % 1) predict xl(n) from xl(n-1)

        % define the prediction
        x_l_hat = xl_l_tilde(nn-1);

        % compute the residual
        d_l = xl(nn) - x_l_hat;
        
        % quantize the residual
        d_l_tilde = delta_l * floor(d_l/delta_l) + delta_l/2;
        
        % reconstruct the signal
        xl_l_tilde(nn) = d_l_tilde + x_l_hat;
        
        % 2) predict xl(n) from xr(n)
        x_r_hat = xr(nn);
        d_r = xl(nn) - x_r_hat;
        d_r_tilde = delta_r * floor(d_r/delta_r) + delta_r/2;
        xl_r_tilde(nn) = d_r_tilde + x_r_hat;

        % 3) predict xl(n) from dummy linear combination
        x_d_hat = 5 * xl_l_tilde(nn);
        d_d = xl(nn) - x_d_hat;
        d_d_tilde = delta_d * floor(d_d/delta_d) + delta_d/2;
        xl_d_tilde(nn) = d_d_tilde + x_d_hat;

    end

    % PCM
    xl_pcm_tilde = delta_pcm * floor(xl/delta_pcm) + delta_pcm/2;

    % MSE
    MSE_l(ii) = mean((xl - xl_l_tilde).^2);
    MSE_r(ii) = mean((xl - xl_r_tilde).^2);
    MSE_d(ii) = mean((xl - xl_d_tilde).^2);
    MSE_pcm(ii) = mean((xl - xl_pcm_tilde).^2);
    
end

% SNR

SNR_l = pow2db(var(xl)./MSE_l);
SNR_r = pow2db(var(xl)./MSE_r);
SNR_d = pow2db(var(xl)./MSE_d);
SNR_pcm = pow2db(var(xl)./MSE_pcm);


%% 2) Compare R-D curves 

figure
plot(R, [SNR_l, SNR_r, SNR_d, SNR_pcm], 'linewidth', 2);
legend('$\hat{x} = xl(n-1)$','$\hat{x} = xr(n)$','$\hat{x} = 5*xl(n)$', 'PCM', 'Interpreter', 'latex');
grid on;
xlabel('Rate [bit/symbol]');
ylabel('SNR [dB]');
set(gca, 'fontsize', 18);

