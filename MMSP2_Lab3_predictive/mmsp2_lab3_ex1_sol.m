%% MMSP2 - Lab 3
%  Exercise 1 - Predictive coding

clear
close all
clc


%% 1) Generate 10000 samples of the random process
%%    x(n) = rho*x(n-1) + z(n), where rho=0.95 and z(n)~N(0,0.1)

rng(21)

N = 10000;
rho = 0.95;
z_var = 0.1;

z = randn(N, 1) * sqrt(z_var);

x = filter([1, -rho], 1, z);

%% 2) Build a PCM codec. Quantize the signal with a uniform quantizer and
%%    R bits. Compute the R-D curve for R=1,2,...,8 bits (in terms of SNR)

R = 1:8;

max_x = max(x);
min_x = min(x);

% % Plot signal distribution
% figure();
% subplot(121);
% plot(x, '.');
% subplot(122);
% bar(hist(x));

MSE_pcm = zeros(length(R), 1);

for ii = 1:length(R)

    % determine delta
    delta = (max_x - min_x) / 2^R(ii);

    % quantize
    x_q = delta * floor(x/delta) + delta/2; % midrise quantizer
    
%     % fix rounding operations
%     x_q_vals = unique(x_q);
%     x_q(x_q == x_q_vals(end)) = x_q_vals(end-1);
    
%     % Plot quantized signal distributions
%     figure();
%     subplot(121);
%     plot(x_q, '.');
%     subplot(122);
%     bar(hist(x_q));

    % compute MSE
    MSE_pcm(ii) = mean((x_q - x).^2);

end

% compute SNR
SNR_pcm = pow2db(var(x)./MSE_pcm);


%% 3) Build a predictive codec in open loop. Use the optimal MMSE predictor.
%%    Use PCM to initialize the codec

% Remark:
% The optimal MMSE predictor is
% x_hat(n) = rho*x(n-1), hence the prediction residual is
% d(n) = x(n) - x_hat(n) = x(n) - rho*x(n-1) = z(n)
% (non-deterministic component)

% For the first sample of the process (i.e. n = 1) we would have
% d(1) = x(1) - rho*x(0), which is impossible to compute.
% So, in order to highlight this fact, we set d(1) = NaN
% Please notice that the value of d(1) will remain unused.
%
% For the second sample of the process (i.e. n = 2) we have
% d(2) = x(2) - rho*x(1) = z(2).
%
% From that remark follows the definition of the vector d as:

d = [NaN; z(2:N)];

max_d = max(d);
min_d = min(d);

MSE_olpc = zeros(length(R), 1);

for ii = 1:length(R)
    
    % we need the quantized signal
    % even if we are in open loop we compute the error on it
    x_tilde = zeros(N, 1);

    % first sample: PCM
    % same steps as before: compute delta and quantize
    delta_pcm = (max_x - min_x) / 2^R(ii);
    x_tilde(1) = delta_pcm * floor(x(1)/delta_pcm) + delta_pcm/2;
    
    % next samples: OLPC
    % same but computed on d instead of x
    delta_olpc = (max_d - min_d) /  2^R(ii);
    d_tilde = delta_olpc * floor(d/delta_olpc) + delta_olpc/2;

    for nn = 2:N
        x_tilde(nn) = d_tilde(nn) + rho * x_tilde(nn-1);
        % rho is our predictor - we assume it is known
        % rho is also known as the receiver
    end

    % MSE
    MSE_olpc(ii) = mean((x - x_tilde).^2);

end

% SNR

SNR_olpc = pow2db(var(x)./MSE_olpc);



%% 4) Build a DPCM codec. Use the optimal MMSE predictor.
%%    Use PCM to initialize the codec.

MSE_dpcm = zeros(length(R), 1);

for ii = 1:length(R)

    x_tilde = zeros(N, 1);

    % first sample: PCM
    delta_pcm = (max_x - min_x) / 2^R(ii);
    x_tilde(1) = delta_pcm * floor(x(1)/delta_pcm) + delta_pcm/2;
    
    % next samples: DPCM
    delta_dpcm = (max_d - min_d) / 2^R(ii);
    % here we assume we know the dynamic range of d
    % in a real case we do not know this -> we should guess delta_dpcm
    
    for nn = 2:N
        % now in computing x_hat we have x_tilde as we are in closed loop
        x_hat = rho * x_tilde(nn-1);
        d = x(nn)-x_hat;
        d_tilde = delta_dpcm * floor(d/delta_dpcm) + delta_dpcm/2;
        x_tilde(nn) = d_tilde + x_hat;
    end

    % MSE
    MSE_dpcm(ii) = mean((x - x_tilde).^2);

end

% SNR
SNR_dpcm = pow2db(var(x)./MSE_dpcm);


%% 5) Compare R-D curves for PCM, open-loop DPCM and closed-loop DPCM


figure()
plot(R, [SNR_dpcm, SNR_olpc, SNR_pcm], 'linewidth', 2);
legend('DPCM Closed Loop','DPCM Open Loop','PCM');
grid on;
xlabel('Rate [bit/symbol]');
ylabel('SNR [dB]');
set(gca, 'fontsize', 18);


