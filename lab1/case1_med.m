function [ class ] = case1_med( x, y, u_A, u_B)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    d_a = -u_A(1) * x -u_A(2)* y + (1/2)*(u_A * u_A');
    d_b = -u_B(1) * x -u_B(2)* y + (1/2)*(u_B * u_B');
    
    if d_a < d_b
        class = 1;
    else
        class =2;
    end
end

