clearvars
close all
clc

%% create auto-regressive signal
% 22-2-17
%% y(n) = rho*y(n-1)+w(n), rho=0.99, w(n) is gaussian with variance=1
rng(21);
N = 500;
rho = 0.99;

% mean = 0, variance=2
var_signal = 2;
w = randn(N,1)*sqrt(var_signal);

% get y
y = zeros(N,1);

y(1) = w(1);
for n = 2:N
    y(n) = rho*y(n-1) + w(n);
end
% clip the values[-15,15]
y(y < -20) = -20;
y(y > 20) = 20;
y = round(y);

figure();
stem(y);
title('y');

%% quantizers
delta_1 = 1;
y1 = delta_1 * floor(y/delta_1) + delta_1/2; % mid-rise
delta_2 = 2;
y2 = delta_2 * floor((y + delta_2/2)/delta_2)+ delta_2/2; % mid-thread
%% custom quantizer
y3 = zeros(length(y),1);
for i=1:y3
    if x<=-5
        y3(i) = -10;
    elseif(x>-5 && x<=5)
          y3(i) = 0;
    else
        y3(i) = 10;
    end
end
disp(y3)

%% plot rate-distortion working point
%% compute entropy

% 
% figure();
% x = [1,2,3];
% y = [1,2,3];
% plot(x,y,'.');
%%

%% KLT
% remove signal mean
x= y1;
x_zm = x - mean(x);

% Estimate the autocorrelation for groups of 8 symbols
N=8;
X_zm = reshape(x_zm,N,length(x_zm)/N);

RR = zeros(N,N,size(X_zm, 2));
for ii = 1:size(X_zm, 2)
    RR(:,:,ii) = X_zm(:,ii)*X_zm(:,ii)';
end
RR_mean = mean(RR,3);

% compute eigenvectors and eigenvalues of the autocorrelation matrix
[V,~] = eig(RR_mean);
% define the transformation matrix 
T_klt = V';
% Klt encoding 
coeff = T_klt*x_zm;
coeff_q = round(coeff);
% Klt decoding 
x_blocks_rec = T_klt'*coeff_q+mean(x);
x_rec = x_blocks_rec(:);
% Klt decoding
mse = mean((x-x_rec).*2);


% DCT
N = 8; % group of symbols length
T_dct = zeros(N,N);
T_dct(1,:) = sqrt(1/N);

l = 1:N;
for k = 2:N
    T_dct(k,:) = sqrt(2/N)*cos(pi/(2*N)*(k-1)*(2.*l-1));
end




