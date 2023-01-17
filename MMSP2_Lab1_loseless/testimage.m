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
%% DCT
B = 16; % block size 16*16
I_1 = double(I(:,:,1));
I_1_dec = zeros(size(I_1));
I_1_enc = zeros(size(I_1));

[H,W] = size(I_1);
for h=1:B:H-B+1
	for w= 1:B:W-B+1
		block = I_1(h:h+B-1,w:w+B-1);
		block_dct = dct2(block);
		block_dct_q = delta.*round(block_dct/delta);
		block_idct = idct2(block_dct_q);
		I_1_enc(h:h+B-1,w:w+B-1) = block_dct_q;
		I_1_dec(h:h+B-1,w:w+B-1) = block_idct;
	end
end

%% PSNR
MSE_1 = mean((I_1(:)-I_1_dec(:)).^2);
PSNR_1 = 20*log10(255/sqrt(MSE_1));

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

