%% MMSP2 - Lab 6
%  Exercise 1 - ME and MC

clear
close all
clc

%% 1) Load the sequence 'table_tennis.mat' consisting of two grayscale frames

load table_tennis.mat

implay(uint8(table_tennis), 5)

%% 2) Select the 8x8 block starting at (x,y)=(35,150) of the second frame.
% Perform ME using W=16 pixels and the first frame as reference

N = 8;
W = 16;
y0 = 35;
x0 = 150;

% select the 8x8 blocks
block = table_tennis(y0:y0+N-1, x0:x0+N-1, 2);

% build cost function testing all possible motion vectors
sad = zeros(2*W+1, 2*W+1); % SAD = sum of absolute differences

for dy = -W:W
    for dx = -W:W

        pred_block = table_tennis(y0+dy:y0+dy+N-1, x0+dx:x0+dx+N-1);
        sad_block = sum(abs(pred_block(:) - block(:)));
        sad(dy+W+1, dx+W+1) = sad_block;

    end
end

% Find the position of the minimum value of the cost function
[~, idx_min] = min(sad(:));
[y1_idx,x1_idx] = ind2sub(size(sad),idx_min);

y_mv = y1_idx-W-1;
x_mv = x1_idx-W-1;

%% 3) Display the cost function, the starting block and its estimate

% Plot the cost function
figure(1);
imagesc(sad);
axis image;
colorbar;

% figure()
% surf(sad)

% Plot the starting block and its estimate
figure(2)
imshow(table_tennis(:,:,1)/255);
hold on
plot([x0 x0+N-1], [y0 y0], 'LineWidth', 2, 'Color', 'red')
plot([x0 x0], [y0 y0+N-1], 'LineWidth', 2, 'Color', 'red')
plot([x0+N-1 x0+N-1], [y0 y0+N-1], 'LineWidth', 2, 'Color', 'red')
% plot([x0 x0+N-1], [y0+N-1 y0+N-1], 'LineWidth', 2, 'Color', 'red')

x1 = x0 + x_mv;
y1 = y0 + y_mv;

plot([x1 x1+N-1],[y1 y1],'LineWidth',2,'Color','green');
plot([x1 x1],[y1 y1+N-1],'LineWidth',2,'Color','green');
plot([x1+N-1 x1+N-1],[y1 y1+N-1],'LineWidth',2,'Color','green');
plot([x1 x1+N-1],[y1+N-1 y1+N-1],'LineWidth',2,'Color','green');


%% Display the two frames as a GIF
implay(uint8(table_tennis), 5);

