function [ class ] = case2_ged( X, Y, u_C, u_D, u_E, sigma_C, sigma_D, sigma_E )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    x = [X;Y];
    u_C = u_C';
    u_D = u_D';
    u_E = u_E';

    d_c = (x - u_C)' * inv(sigma_C) * (x-u_C);
    d_d = (x - u_D)' * inv(sigma_D) * (x-u_D);
    d_e = (x - u_E)' * inv(sigma_E) * (x-u_E);

    [m, class] = min([d_c, d_d, d_e]);

end

