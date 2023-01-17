%% MMSP2 - Lab 1
%  Exercise 4 - Discrete memoryless source coding
%  From 19 June 2006 exam

clearvars
close all
clc

%% Generate N=10000 samples of an AR(1) random process
%% y(n) = rho*y(n-1)+w(n), rho=0.99, w(n) is gaussian with variance=1


rng(21);
N = 1e4;
rho = 0.99;
w_std = 1;

w = randn(N,1)*w_std;

% we have two different ways to find x(n) (use only one of them):

% 1)first way: filter function

A =[1, -rho];

B = 1;
yy = filter(B, A, w);

% second way: for loop

y = zeros(N,1);

y(1) = w(1);
for n = 2:N
    y(n) = rho*y(n-1) + w(n);
end


%% Clip sample values in the range [-20,20] and round to nearest integer

y(y < -20) = -20;
y(y > 20) = 20;
y = round(y);

figure();
stem(y);
title('y');

%% Compute H(Y) assuming that Y is a discrete memoryless source

alphabet = unique(y);

d = hist(y,alphabet);
p = d/sum(d);

H = -sum(p(d>0) .* log2(p(d>0)));

% compare H(y) with the maximum entropy of a source having the same
% alphabet

M = log2(length(alphabet));

fprintf('entropy of x: %.3f bit/symbol <= %.2f bit/symbol\n',H, M);




