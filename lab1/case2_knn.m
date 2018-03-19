function [ class ] = case1_knn( X, Y , A, B ,C)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    min_distance_class1 = inf(1, 5);
    min_distance_class2 = inf(1, 5);
    min_distance_class3 = inf(1, 5);
    for i=1:size(A, 1)
        d = ((A(i, 1)-X)^2+ (A(i, 2)-Y)^2);
        [d_max, idx] = max(min_distance_class1);
        if d < d_max
            min_distance_class1(idx) = d;
        end
    end
    for i=1:size(B, 1)
        d = ((B(i, 1)-X)^2+ (B(i, 2)-Y)^2);
        [d_max, idx] = max(min_distance_class2);
        if d < d_max
            min_distance_class2(idx) = d;
        end
    end
    for i=1:size(C, 1)
        d = ((C(i, 1)-X)^2+ (C(i, 2)-Y)^2);
        [d_max, idx] = max(min_distance_class3);
        if d < d_max
            min_distance_class3(idx) = d;
        end
    end
    
    mean_class1 = mean(min_distance_class1);
    mean_class2 = mean(min_distance_class2);
    mean_class3 = mean(min_distance_class3);
    if mean_class1 <= mean_class2 && mean_class1 <= mean_class3
        class = 1;
    else
        if mean_class2 <= mean_class1 && mean_class2 <= mean_class3
            class = 2;
        else
            class = 3;
        end
    end
end

