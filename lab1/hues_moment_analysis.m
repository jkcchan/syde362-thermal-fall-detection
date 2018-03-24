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

scatter(falling_area_C1, falling_bb_ratio_C1);
hold on;
scatter(falling_area_C2, falling_bb_ratio_C2);
hold on;

%%
% MED for Case 1

a = linspace(810, 2914,90);
b = linspace(0, 4, 90);
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
for i=1:size(area, 1)
    gt = 1;
    pred = case1_med(area(i), bb_ratio(i), mean_C1, mean_C2);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(bb_ratio, 1)
    gt = 2;
    pred = case1_med(area(i), bb_ratio(i), mean_C1, mean_C2);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on
