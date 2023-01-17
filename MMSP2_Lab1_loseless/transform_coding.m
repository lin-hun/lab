function [x_tilde, R_opt,snr] = transform_coding(x,T,R)
%TRANSFORM_CODING Folding, projection, quantization, inverse projection,
%unfolding with Optimal Bit Allocation
%   x: 1D input signal
%   T: transformation matrix (square matrix, each row is a basis)
%   R: bits per symbols budget

% Determine group of symbols length based on transformation length
N = size(T,1);

% reshaping (possibly cutting) the signal in order to have groups of symbols
x = x(1:floor(length(x)/N)*N);
X = reshape(x, N, length(x)/N);

% apply the transformation
Y = T*X;

% compute optimal bit allocation
var_Y = var(Y,0,2);
R_opt = round(R+0.5*log2(var_Y./geomean(var_Y)));

% compute delta for each coefficient
delta = (max(Y,[],2)-min(Y,[],2))./(2.^R_opt);

% quantize Y
Y_tilde = floor(Y./delta).*delta + delta/2;

% invert the transformation
X_tilde = T'*Y_tilde;
x_tilde = X_tilde(:);

% compute SNR
mse = mean((x-x_tilde).^2);
snr = 10*log10(var(x)/mse);

end

