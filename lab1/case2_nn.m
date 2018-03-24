function [ class ] = case2_nn( X, Y , C, D,E )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    min_distance = inf;
    class = 0;
    for i=1:size(C, 1)
        d = ((C(i, 1)-X)^2+ (C(i, 2)-Y)^2);
        if d < min_distance
            min_distance = d;
            class = 1;
        end
    end
    for i=1:size(D, 1)
        d = ((D(i, 1)-X)^2+ (D(i, 2)-Y)^2);
        if d < min_distance
            min_distance = d;
            class = 2;
        end
    end
    
    for i=1:size(E, 1)
        d = ((E(i, 1)-X)^2+ (E(i, 2)-Y)^2);
        if d < min_distance
            min_distance = d;
            class = 3;
        end
    end
end

