function [ class ] = case2_med( x, y, u_C, u_D, u_E )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    d_c = (x-u_C(1))^2+ (y-u_C(2))^2;
    d_d = (x-u_D(1))^2+ (y-u_D(2))^2;
    d_e = (x-u_E(1))^2+ (y-u_E(2))^2;
   

    if d_c < d_d && d_c < d_e
        class = 1;
    elseif d_d < d_c && d_d < d_e
        class = 2;
    else
        class =3;
    end
    %[foo, class] = min([d_a, d_b, d_c]);
    

end

