function [ class ] = case1_ged( X, Y , u_A, u_B, sigma_A, sigma_B )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    x = [X;Y];
    u_A = u_A';
    u_B = u_B';

    d_a = (x - u_A)' * inv(sigma_A) * (x-u_A);
    d_b = (x - u_B)' * inv(sigma_B) * (x-u_B);
    
    if d_a < d_b
        class =1;
    else
        class=2;
    end
end

