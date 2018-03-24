function [ class ] = case1_nn( X, Y , A, B )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    min_distance = inf;
    class = 0;
    for i=1:size(A, 1)
        d = ((A(i, 1)-X)^2+ (A(i, 2)-Y)^2);
        if d < min_distance
            min_distance = d;
            class = 1;
        end
    end
    for i=1:size(B, 1)
        d = ((B(i, 1)-X)^2+ (B(i, 2)-Y)^2);
        if d < min_distance
            min_distance = d;
            class = 2;
        end
    end
end

