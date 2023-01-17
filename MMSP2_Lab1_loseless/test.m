clearvars;
close all;
clc;

rng(21);

%% generate samples
x_var = 3;
x = randn(1000,1) * sqrt(x_var);

%% mid-rise
delta = 2;
y1 = delta*(floor(x/delta)+1/2); %% won't pass zero 

figure;
plot(x,y1,'.');
%hist(y1);
title('mid-rise');

%% mid-tread
delta = 2;
y2 = delta * floor(x/delta+1/2);

figure();
plot(x,y2,'.')
title('mid-tread');


%% M=4 mid-tread
M = 4;
delta = (max(x)-min(x))/M;
y3 = delta * floor(x/delta+1/2);

%% rounding y3
y3_values = unique(y3);
y3(y3==y3_values(end)) = y3_values(end-1);

figure();
plot(x,y3,'.')
grid on;
title('when M = 4')

%% quantize cb as production level(y)
%% th as thresholds(x)
cb = [-5,-3,-1,0,1,3,5];
th = [-4,-2,-0.5,0.5,2,4];

th = [-inf, th, inf];
y4 = zeros(size(x));

for level = 1:length(cb)
    mask = x>=th(level) & x<th(level+1);
    y4(mask) = cb(level);
end

figure();
plot(x,y4,'.'); %% if use th, there is an error, not the same length
title('fix cb & th');

%% power and var
e = y1-x;

Px = mean(x.^2);
Pe = mean(e.^2);

SNR = Px/Pe;

%% distortion for every quantizer
mse1 = mean((y1-x).^2);
mse2 = mean((y2-x).^2);
mse3 = mean((y3-x).^2);
mse4 = mean((y4-x).^2);

%% SNR
snr1 = mean(x.^2)/mse1;
snr2 = mean(x.^2)/mse2;
snr3 = mean(x.^2)/mse3;
snr4 = mean(x.^2)/mse4;

fprintf('snr1:%d',snr1);

%% convert to db
snr1_db = pow2db(snr1);
disp(snr1_db);
