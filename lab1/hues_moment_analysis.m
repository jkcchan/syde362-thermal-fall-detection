filename = '../regression_2.csv';
M = csvread(filename, 1, 0);

% 0 is not falling
% 1 is falling
is_falling = M(:, 3);
% 1 is standing -> 0 in is_falling
% 2 is sitting down -> 0 in is_falling
% 3 is laying down -> 1 in is_falling
position = M(:, 11);

area = M(:, 1);
bb_ratio = M(:, 2);
hue_variance = M(:, 12);

falling_area_C1 = [];
falling_area_C2 = [];

falling_bb_ratio_C1 = [];
falling_bb_ratio_C2 = [];

falling_hue_variance_C1 = [];
falling_hue_variance_C2 = [];

count1 = 1;
count2 = 1;

for i=1:size(M,1)
    if (is_falling(i) == 1)
        falling_area_C1(count1) = area(i);
        falling_bb_ratio_C1(count1) = bb_ratio(i);
        falling_hue_variance_C1(count1) = hue_variance(i);
        count1 = count1 + 1;
    else
        falling_area_C2(count2) = area(i);
        falling_bb_ratio_C2(count2) = bb_ratio(i);
        falling_hue_variance_C2(count2) = hue_variance(i);
        count2 = count2 + 1;
    end
end

mean_falling_area_C1 = mean(falling_area_C1);
mean_falling_area_C2 = mean(falling_area_C2);

mean_bb_ratio_C1 = mean(falling_bb_ratio_C1);
mean_bb_ratio_C2 = mean(falling_bb_ratio_C2);

mean_C1 = [mean_falling_area_C1, mean_bb_ratio_C1];
mean_C2 = [mean_falling_area_C2, mean_bb_ratio_C2];

sigma_C1 = cov(falling_area_C1, falling_bb_ratio_C1);
sigma_C2 = cov(falling_area_C2, falling_bb_ratio_C2);

n_C1 = length(falling_area_C1);
n_C2 = length(falling_area_C2);

scatter(falling_area_C1, falling_bb_ratio_C1);
hold on;
scatter(falling_area_C2, falling_bb_ratio_C2);
hold on;

%%
% MED for Case 1

a = linspace(810, 2914,300);
b = linspace(0, 4, 300);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));
for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case1_med(X(i,j), Y(i,j), mean_C1, mean_C2);
    end
end
s = contour(X, Y, classes, 'r');


cfn_matrix = [0, 0;
              0, 0];
for i=1:size(falling_area_C1, 2)
    gt = 1;
    pred = case1_med(falling_area_C1(i), falling_bb_ratio_C1(i), mean_C1, mean_C2);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(falling_area_C2, 2)
    gt = 2;
    pred = case1_med(falling_area_C2(i), falling_bb_ratio_C2(i), mean_C1, mean_C2);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on

%%
% GED for Case 1
a = linspace(810, 2914,300);
b = linspace(0, 4, 300);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case1_ged(X(i,j), Y(i,j), mean_C1, mean_C2, sigma_C1, sigma_C2);
    end
end

s = contour(X, Y, classes,'g');

cfn_matrix = [0, 0;
              0, 0];
 for i=1:size(falling_area_C1, 2)
     gt = 1;
     pred = case1_ged(falling_area_C1(i), falling_bb_ratio_C1(i), mean_C1, mean_C2, sigma_C1, sigma_C2);
     cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
 end
 
for i=1:size(falling_area_C2, 2)
    gt = 2;
    pred = case1_ged(falling_area_C2(i), falling_bb_ratio_C2(i), mean_C1, mean_C2, sigma_C1, sigma_C2);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on

%%
% MAP for Case 1
a = linspace(810, 2914,300);
b = linspace(0, 4, 300);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case1_map(X(i,j), Y(i,j),  mean_C1, mean_C2, sigma_C1, sigma_C2, n_C1/(n_C1+n_C2), n_C2/(n_C1+n_C2));
    end
end

s = contour(X, Y, classes,'b');
cfn_matrix = [0, 0;
              0, 0];
for i=1:size(falling_area_C1, 2)
    gt = 1;
    pred = case1_map(falling_area_C1(i), falling_bb_ratio_C1(i), mean_C1, mean_C2, sigma_C1, sigma_C2, n_C1/(n_C1+n_C2), n_C2/(n_C1+n_C2));
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(falling_area_C2, 2)
    gt = 2;
    pred = case1_map(falling_area_C2(i), falling_bb_ratio_C2(i), mean_C1, mean_C2, sigma_C1, sigma_C2, n_C1/(n_C1+n_C2), n_C2/(n_C1+n_C2));
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on

%%
% NN Case 1
% Have to run sperately because ratio is scaled by 1000
figure
scatter(falling_area_C1, falling_bb_ratio_C1*1000);
hold on;
scatter(falling_area_C2, falling_bb_ratio_C2*1000);
hold on;

C1 = [falling_area_C1; falling_bb_ratio_C1*1000]';
C2 = [falling_area_C2; falling_bb_ratio_C2*1000]';

a = linspace(500, 3000, 1000);
b = linspace(0, 4000, 1000);

[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case1_nn(X(i,j), Y(i,j), C1, C2);
    end
end

s = contour(X, Y, classes, 'r');

%%
% KNN Case 1
a = linspace(500, 3000, 1000);
b = linspace(0, 4000, 1000);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case1_knn(X(i,j), Y(i,j), C1, C2);
    end
end

s = contour(X, Y, classes, 'g');
hold on