%% MMSP2 - Lab 2
%  Exercise 4 - Vector quantization

clearvars
close all
clc

%% 1) Consider a 2D vector quantizer with codebook y1=(1,2), y2=(1,4), y3=(-1,2), y4=(0,-2)
%%    Show optimal assignement regions

cb = [ 1,1,-1,0;
       2,4,2,-2 ];

figure()
voronoi(cb(1,:),cb(2,:));
hold on;
plot(cb(1,:),cb(2,:),'o', 'markersize', 12, 'linewidth', 2);
grid on;
xlim([min(cb(1,:))-1,max(cb(1,:))+1]);
ylim([min(cb(2,:))-1,max(cb(2,:))+1]);
set(gca, 'fontsize', 18);

%% 2) Quantize the sequence x=(-4:5) using groups of 2 consecutive 
% samples at time

x = -4:5;

xgroup = [x(1:2:end); x(2:2:end)];

xgroup_q = zeros(size(xgroup));

for sample_idx = 1:size(xgroup,2)

    sample = xgroup(:,sample_idx);
    
    % compute the euclidean distance from each centroid
    dist = sum((sample - cb) .^2);
    
    % find the nearest centroid
    [~,min_idx] = min(dist);
    sample_code = cb(:,min_idx);
    
    xgroup_q(:,sample_idx) = sample_code;
    
end

% MATLAB way
vqenc = dsp.VectorQuantizerEncoder('Codebook',cb);
xgroup_q_ml_idxs = vqenc.step(xgroup);
xgroup_q_ml = cb(:,xgroup_q_ml_idxs+1);

%% 3) Plot the original and the quantized sequences

figure();
voronoi(cb(1,:),cb(2,:));
hold on;
leg = [];
leg(1) = plot(xgroup(1,:),xgroup(2,:),'x', 'markersize', 12, 'linewidth', 2);
leg(2) = plot(xgroup_q(1,:),xgroup_q(2,:),'o', 'markersize', 12, 'linewidth', 2);
leg(3) = plot(xgroup_q_ml(1,:),xgroup_q_ml(2,:),'p', 'markersize', 12, 'linewidth', 2);
legend(leg,'Original','Quantized', 'Quantized ML');
grid on;
set(gca, 'fontsize', 18);