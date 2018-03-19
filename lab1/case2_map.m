function [ class ] = case2_map(  X, Y, u_C, u_D, u_E, sigma_C, sigma_D, sigma_E , Prior_C, Prior_D, Prior_E)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    x = [X;Y];
    u_C = u_C';
    u_D = u_D';
    u_E = u_E';

    d_c = (x - u_C)' * inv(sigma_C) * (x-u_C);
    d_d = (x - u_D)' * inv(sigma_D) * (x-u_D);
    d_e = (x - u_E)' * inv(sigma_E) * (x-u_E);
    
    thresh_CD = 2*log(Prior_D/Prior_C) + log(det(sigma_C)/det(sigma_D));
    
    thresh_CE = 2*log(Prior_E/Prior_C) + log(det(sigma_C)/det(sigma_E));
    thresh_ED = 2*log(Prior_D/Prior_E) + log(det(sigma_E)/det(sigma_D));

    if d_d - d_c < thresh_CD
        pred_1 = 2;
    else
        pred_1 = 1;
    end
    
    if d_e - d_c < thresh_CE
        pred_2 = 3;
    else
        pred_2 = 1;
    end
    
    if d_d - d_e < thresh_ED
        pred_3 = 2;
    else
        pred_3 = 3;
    end
    
    class = mode([pred_1, pred_2, pred_3]);
    
end

