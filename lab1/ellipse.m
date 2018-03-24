function [ x, y ] = ellipse( V, D , u)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
x_center = u(1);
y_center = u(2);
a = sqrt(D(1,1));
b = sqrt(D(2,2));
t = 0 : 0.01 : 2*pi;

theta = atan(V(2, 1)/V(1,1));

x = a*cos(t)*cos(theta) - b*sin(t)*sin(theta) + x_center;
y = a*cos(t)*sin(theta) + b*sin(t)*cos(theta) + y_center;

end

