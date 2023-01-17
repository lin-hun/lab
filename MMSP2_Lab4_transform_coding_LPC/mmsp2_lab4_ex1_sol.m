%% MMSP2 - Lab 4
%  Exercise 1 - Transform coding

clear
close all
clc

%% 1) Load the first 4s of the file 'gb.wav' and quantize it with PCM and R=8 bit.
%%    Compute the MSE and perceptually evaluate the result.
[x, Fs] = audioread('gb.wav');

R = 8;
len = 4;
x = x(1:len*Fs);

delta_pcm = (max(x)-min(x))/(2^R);
x_pcm = delta_pcm*floor(x/delta_pcm) + delta_pcm/2;

mse_pcm = mean((x-x_pcm).^2);
snr_pcm = 10*log10(var(x)/mse_pcm);

disp(['SNR PCM: ' num2str(snr_pcm) 'dB' ]);
% Listen the audio track 
% sound(x, Fs)
% sound(x_pcm, Fs)


%% 2) Consider groups of 8 symbols and quantize them using an optimal allocation of the 8 bits
% complete the function transform_coding
% hint: consider each block of 8 symbols as if composed by 8 transform
% coefficients

N = 8; %group of symbols length

% Define the transformation matrix
T_eye = eye(N);
[x_tilde_eye, R_eye,snr_eye] = transform_coding(x,T_eye,R);

disp(['SNR Eye: ' num2str(snr_eye) 'dB' ]);

%% 3) Consider DCT transformation and repeat step 2 over transformed 
%%    coefficients. Find the distortion and evaluate the perceived quality.

N = 8; %group of symbols length

% Define the transformation matrix
T_dct = zeros(N,N);
T_dct(1,:) = sqrt(1/N);

l = 1:N;
for k = 2:N
    T_dct(k,:) = sqrt(2/N)*cos(pi/(2*N)*(k-1)*(2.*l-1));
end

%see also: T = dctmtx(N);

[x_tilde_dct, R_dct,snr_dct] = transform_coding(x,T_dct,R);

disp(['SNR DCT: ' num2str(snr_dct) 'dB' ]);

%% 4) Consider a Karhunen-Loeve transformation and repeat step 2 over transformed 
%%    coefficients. Find the distortion and evaluate the perceived quality.

% compute correlation
% hint: remove the average from the signal
% hint: compute many 8x8 correlation matrix, then average them

% remove signal mean
x_zm = x - mean(x);

% Estimate the autocorrelation for groups of 8 symbols
X_zm = reshape(x_zm,N,length(x_zm)/N);

RR = zeros(N,N,size(X_zm, 2));
for ii = 1:size(X_zm, 2)
    RR(:,:,ii) = X_zm(:,ii)*X_zm(:,ii)';
end
RR_mean = mean(RR,3);

% compute eigenvectors and eigenvalues of the autocorrelation matrix
[V,~] = eig(RR_mean);

% define the transformation matrix (be aware that we need a transformation
% matrix where the rows are our projection basis!)
T_klt = V';

[x_tilde_klt, R_klt,snr_klt] = transform_coding(x_zm,T_klt,R); % of course if we want to go back to x we need to add the mean back

disp(['SNR KLT: ' num2str(snr_klt) 'dB' ]);


%% 5) Plot the amount of bits allocated for each of the proposed solutions

figure();
subplot(3,1,1);
bar(R_eye);
xlabel('Coefficient');
ylabel('bit');
title('Eye');
grid on;
set(gca, 'fontsize', 18);

subplot(3,1,2);
bar(R_dct);
xlabel('Coefficient');
ylabel('bit');
title('DCT');
grid on;
set(gca, 'fontsize', 18);

subplot(3,1,3);
bar(R_klt);
xlabel('Coefficient');
ylabel('bit');
title('KLT');
grid on;
set(gca, 'fontsize', 18);