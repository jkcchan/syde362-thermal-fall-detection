function [ class ] = case1_map( X, Y , u_A, u_B, sigma_A, sigma_B, Prior_A, Prior_B)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    x = [X;Y];
    u_A = u_A';
    u_B = u_B';
    d_a = (x - u_A)' * inv(sigma_A) * (x-u_A);
    d_b = (x - u_B)' * inv(sigma_B) * (x-u_B);
    
    thresh = 2*log(Prior_B/Prior_A) + log(det(sigma_A)/det(sigma_B));
    
    if d_b - d_a < thresh
        class = 2;
    else
        class=1;
    end
    
    
end

