clearvars;
close all;
clc;

% load image & show
im = imread('peppers.png');
figure();
imagesc(im), axis image

% extract green plane
im = double(im);
G = im(:, :, 2);

% PCM
%% 1.define delta 2.quantize(any)
% build uniform quantizer and R bits.
R = 4;
% vectorize the signals
x = G(:);
% max & min
max_x = max(x);
min_x = min(x);

% determine delta
delta = (max_x - min_x) / 2^R;

% quantize
x_q = delta * floor(x/delta) + delta/2; % midrise quantizer, doesnot include 0
% delta * floor((x + delta/2)/delta)+ delta/2; % mid-tread

% for ii = 1:length(R) % if R = [1,2,3,4,5,6] need to for

% end

% compute entropy
alphabet = 0:255;
d_G = hist(x_q(:), alphabet);
% pmf
p_G = d_G/sum(d_G);
p_G = p_G(d_G>0);
H_G = -sum(p_G .* log2(p_G));
disp(['H of quantized result=',num2str(H_G)]);

% SNR SNR = Px/Pe 
% pow2db(snr) (recommend way)
mse = mean((x_q(:)-x).^2);
snr = var(x) / mse; % x original signal, 
snr2_db = pow2db(snr);
disp(['snr:',num2str(snr)]);
disp(['snr dB:',num2str(snr2_db)])

%% transform coding()

%% DCT: set of cosine computed(the first cos is special, others are same)
%% KLT: auto-correlation computation
%% DCT: