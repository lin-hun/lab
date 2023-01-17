clearvars;
close all;
clc;

% read image & display
im = imread('peppers.png');
% display image
% imshow(im)
% imagesc(im)

% read RGB plane
im = double(im);
R = im(:, :, 1);
G = im(:, :, 2);
B = im(:, :, 3);

condition  = "YCbCr";
% compute & display histogram 

% compute  entropy H(R), H(G), H(B)
%% 1. define alphabet
%% 2. compute distribution
%% 3. compute p.m.f
%% 4. compute entropy
if condition == "entropy"
    R = R(:);
    
    alphabet = 0:255;% unique
    d_R = hist(R, alphabet);
    % pmf
    p_R = d_R/sum(d_R);
    
    p_R = p_R(d_R>0);
    % entropy
    H_R = -sum(p_R .* log2(p_R));
    disp(['H Red channel = ', num2str(H_R)]);
end
% PCM(pulse code modulation)
if condition == "PCM"
    %% 1.define delta 2.quantize(any)
    % build uniform quantizer and R bits.
    R = 4;
    % vectorize the signals
    x = G(:);
    % max & min
    max_x = max(x);
    min_x = min(x);

    MSE_pcm = zeros(length(R), 1);

    for ii = 1:length(R)
        % determine delta
        delta = (max_x - min_x) / 2^R(ii);
    
        % quantize
        x_q = delta * floor(x/delta) + delta/2; % midrise quantizer, doesnot include 0
        % delta * floor((x + delta/2)/delta)+ delta/2; % mid-tread
        
        % compute MSE
        MSE_pcm(ii) = mean((x_q - x).^2);
    end

end
% transform coding
%% 1.simple PCM quantization after applying a transform(transform coding)
%% 2.the transform is applied to leverage(affect) the characteristics of the signal
%% 3.we want to reduce the variance of the signal itself
%%
N = 8; % dimension of the image block

K = 8; % dimension of the projection space

block = Y(1:N, 1:N);
block_dct = dct2(block);
g = zeros(N,N,K,K);
for n1 = 1:N
    for n2 = 1:N
        for k1=1:K
            for k2=1:K
                
                if k1 == 1
                    a_k1 = sqrt(1/N);
                else
                    a_k1 = sqrt(2/N);
                end
    
                if k2 == 1
                    a_k2 = sqrt(1/N);
                else 
                    a_k2 = sqrt(2/N);
                end
                
                g(n1,n2,k1,k2) = a_k1*cos((2*(n1-1)+1)*pi*(k1-1)/(2*N))*a_k2*cos((2*(n2-1)+1)*pi*(k2-1)/(2*N));

                block_dct(k1,k2,4) = block_dct(k1,k2,4) + g(n1,n2,k1,k2)*block(n1,n2);
            end
        end
    end
end
% Compute the MSE between coefficients in the different cases
block_mse = zeros(4,4);
for idx1 = 1:4
    for idx2 = 1:4
        block_mse(idx1,idx2) = (sum(sum((block_dct(:,:,idx1)-block_dct(:,:,idx2)).^2)))/(N.^2);
    end
end

disp('MSE');
disp(block_mse);

%% JPEG 
%% 1.define image block
%% 2.compute DCT(remember to subtract the level offset)
%% 3.quantize and multiply by jpeg matrix
%% 4.reconstruct the image(iDCT)
if condition == "YCbCr"
    % RGB 2 YCbCr(用第一行的数据,只能得到Y,亮度)
    Y = 0.299*R + 0.587*G + 0.144*B;
    figure();
    imagesc(Y), axis image
    % DCT 8x8 block
    % we only consider the first image block
    block = Y(1:N, 1:N);

    block_dct = zeros(K, K, 4);

end

% SNR SNR = Px/Pe 
% pow2db(snr) (recommend way)
mse = mean((x_q(:)-x).^2);
snr = var(x) / mse; % x original signal, 
snr2_db = pow2db(snr);
disp(['snr:',num2str(snr)]);
disp(['snr dB:',num2str(snr2_db)])

