%%
% Plot case 1
t = linspace(0,2*pi);

N_A = 200;
N_B = 200;
u_A = [5, 10];
u_B = [10, 15];
sigma_A = [8, 0;
           0, 4];

sigma_B = [8, 0;
           0, 4];
       
  
rand_A = randn(N_A,2)*chol(sigma_A)  + repmat(u_A,N_A,1);
rand_B = randn(N_B,2)*chol(sigma_B)  + repmat(u_B,N_B,1);


%P(e) calculation
errors=0;
for i=1:size(rand_A, 1)
    x = [rand_A(i, 1); rand_A(i, 2)];
    
    p_A = (1/((2*pi)^0.5 *det(sigma_A)^0.5))*exp(-0.5*((x-u_A')'*inv(sigma_A)*(x-u_A')));
    p_B = (1/((2*pi)^0.5 *det(sigma_B)^0.5))*exp(-0.5*((x-u_B')'*inv(sigma_B)*(x-u_B')));

    if p_B > p_A
        errors = errors + 1; 
    end 
end
for i=1:size(rand_B, 1)
    x = [rand_B(i, 1); rand_B(i, 2)];
    
    p_A = (1/((2*pi)^0.5 *det(sigma_A)^0.5))*exp(-0.5*((x-u_A')'*inv(sigma_A)*(x-u_A')));
    p_B = (1/((2*pi)^0.5 *det(sigma_B)^0.5))*exp(-0.5*((x-u_B')'*inv(sigma_B)*(x-u_B')));

    if p_A > p_B
        errors = errors + 1; 
    end 
end

p_e = errors/(N_A+N_B)


scatter(rand_A(:,1), rand_A(:,2));
hold on
u = u_A;
sigma = sigma_A;
plot(sqrt(sigma(1,1))*cos(t)+u(1),sqrt(sigma(2,2))*sin(t)+u(2));

scatter(rand_B(:,1), rand_B(:,2));
hold on
u = u_B;
sigma = sigma_B;
plot(sqrt(sigma(1,1))*cos(t)+u(1),sqrt(sigma(2,2))*sin(t)+u(2));

hold on
%%
% Plot case 2


% standing
% lying
% sitting
% x mean, y mean, covariance
% 360.5 294.333333333 -311
% 405.8 212.85 925
% 235.222222222 287.666666667 -836

N_C = 50;
N_D = 50;
N_E = 50;

u_C = [360.5, 294.33];
u_D = [405.8, 212.85];
u_E = [235.222, 287.666];


sigma_C = [1433.916, -311;
           -311, 801.888888889];
sigma_D = [1457.06, 925;
           925, 2690.1275];
sigma_E = [3910.83950617, -836;
           -836, 1053.77777778];
       
rand_C = randn(N_C,2)*chol(sigma_C)  + repmat(u_C,N_C,1);
rand_D = randn(N_D,2)*chol(sigma_D)  + repmat(u_D,N_D,1);
rand_E = randn(N_E,2)*chol(sigma_E)  + repmat(u_E,N_E,1);

%P(e) calculation
errors=0;
for i=1:size(rand_C, 1)
    x = [rand_C(i, 1); rand_C(i, 2)];
    
    p_C = (1/((2*pi)^0.5 *det(sigma_C)^0.5))*exp(-0.5*((x-u_C')'*inv(sigma_C)*(x-u_C')));
    p_D = (1/((2*pi)^0.5 *det(sigma_D)^0.5))*exp(-0.5*((x-u_D')'*inv(sigma_D)*(x-u_D')));
    p_E = (1/((2*pi)^0.5 *det(sigma_E)^0.5))*exp(-0.5*((x-u_E')'*inv(sigma_E)*(x-u_E')));

    if p_D > p_C || p_E > p_C
        errors = errors + 1; 
    end 
end
for i=1:size(rand_D, 1)
    x = [rand_D(i, 1); rand_D(i, 2)];
    
    p_C = (1/((2*pi)^0.5 *det(sigma_C)^0.5))*exp(-0.5*((x-u_C')'*inv(sigma_C)*(x-u_C')));
    p_D = (1/((2*pi)^0.5 *det(sigma_D)^0.5))*exp(-0.5*((x-u_D')'*inv(sigma_D)*(x-u_D')));
    p_E = (1/((2*pi)^0.5 *det(sigma_E)^0.5))*exp(-0.5*((x-u_E')'*inv(sigma_E)*(x-u_E')));

    if p_C > p_D || p_E > p_D
        errors = errors + 1; 
    end 
end
for i=1:size(rand_E, 1)
    x = [rand_E(i, 1); rand_E(i, 2)];
    
    p_C = (1/((2*pi)^0.5 *det(sigma_C)^0.5))*exp(-0.5*((x-u_C')'*inv(sigma_C)*(x-u_C')));
    p_D = (1/((2*pi)^0.5 *det(sigma_D)^0.5))*exp(-0.5*((x-u_D')'*inv(sigma_D)*(x-u_D')));
    p_E = (1/((2*pi)^0.5 *det(sigma_E)^0.5))*exp(-0.5*((x-u_E')'*inv(sigma_E)*(x-u_E')));

    if p_D > p_E || p_C > p_E
        errors = errors + 1; 
    end 
end

p_e = errors/(N_C+N_D+N_E)




scatter(rand_C(:,1), rand_C(:,2))
hold on
[V, D] = eig(sigma_C);
[x, y] = ellipse(V, D, u_C);
plot(x,y)

scatter(rand_D(:,1), rand_D(:,2))
hold on
[V, D] = eig(sigma_D);
[x, y] = ellipse(V, D, u_D);
plot(x,y)

scatter(rand_E(:,1), rand_E(:,2))
hold on
[V, D] = eig(sigma_E);
[x, y] = ellipse(V, D, u_E);
plot(x,y)

hold on
%%
% MED for Case 1

a = linspace(-5, 20,100);
b = linspace(0, 25,100);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));
for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case1_med(X(i,j), Y(i,j), u_A, u_B);
    end
end
s = contour(X, Y, classes, 'r');


cfn_matrix = [0, 0;
              0, 0];
for i=1:size(rand_A, 1)
    gt = 1;
    pred = case1_med(rand_A(i, 1), rand_A(i, 2), u_A, u_B);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_B, 1)
    gt = 2;
    pred = case1_med(rand_B(i, 1), rand_B(i, 2), u_A, u_B);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on

%%
% MED for Case 2

a = linspace(100, 500,1000);
b = linspace(100, 400,1000);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case2_med(X(i,j), Y(i,j), u_C, u_D, u_E);
    end
end

s = contour(X, Y, classes,'r');

cfn_matrix = [0, 0, 0;
              0, 0, 0;
              0, 0, 0];
for i=1:size(rand_C, 1)
    gt = 1;
    pred = case2_med(rand_C(i, 1), rand_C(i, 2), u_C, u_D, u_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_D, 1)
    gt = 2;
    pred = case2_med(rand_D(i, 1), rand_D(i, 2), u_C, u_D, u_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_E, 1)
    gt = 3;
    pred = case2_med(rand_E(i, 1), rand_E(i, 2), u_C, u_D, u_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix

hold on

%%
% GED for Case 1
a = linspace(-5, 20,100);
b = linspace(0, 25,100);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case1_ged(X(i,j), Y(i,j), u_A, u_B, sigma_A, sigma_B);
    end
end

s = contour(X, Y, classes,'g');

cfn_matrix = [0, 0;
              0, 0];
for i=1:size(rand_A, 1)
    gt = 1;
    pred = case1_ged(rand_A(i, 1), rand_A(i, 2), u_A, u_B, sigma_A, sigma_B);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_B, 1)
    gt = 2;
    pred = case1_ged(rand_B(i, 1), rand_B(i, 2),  u_A, u_B, sigma_A, sigma_B);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on
%%
% GED for Case 2
a = linspace(100, 500,1000);
b = linspace(100, 400,1000);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case2_ged(X(i,j), Y(i,j), u_C, u_D, u_E, sigma_C, sigma_D, sigma_E);
    end
end

s = contour(X, Y, classes,'g');
cfn_matrix = [0, 0, 0;
              0, 0, 0;
              0, 0, 0];
for i=1:size(rand_C, 1)
    gt = 1;
    pred = case2_ged(rand_C(i, 1), rand_C(i, 2), u_C, u_D, u_E, sigma_C, sigma_D, sigma_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_D, 1)
    gt = 2;
    pred = case2_ged(rand_D(i, 1), rand_D(i, 2), u_C, u_D, u_E, sigma_C, sigma_D, sigma_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_E, 1)
    gt = 3;
    pred = case2_ged(rand_E(i, 1), rand_E(i, 2), u_C, u_D, u_E, sigma_C, sigma_D, sigma_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on
%%
% MAP for Case 1
a = linspace(-5, 20,100);
b = linspace(0, 25,100);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case1_map(X(i,j), Y(i,j), u_A, u_B, sigma_A, sigma_B, N_A/(N_A+N_B), N_B/(N_A+N_B));
    end
end

s = contour(X, Y, classes,'b');
cfn_matrix = [0, 0;
              0, 0];
for i=1:size(rand_A, 1)
    gt = 1;
    pred = case1_map(rand_A(i, 1), rand_A(i, 2), u_A, u_B, sigma_A, sigma_B, N_A/(N_A+N_B), N_B/(N_A+N_B));
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_B, 1)
    gt = 2;
    pred = case1_map(rand_B(i, 1), rand_B(i, 2),  u_A, u_B, sigma_A, sigma_B, N_A/(N_A+N_B), N_B/(N_A+N_B));
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on
%%
% MAP for Case 2
a = linspace(100, 500,1000);
b = linspace(100, 400,1000);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case2_map(X(i,j), Y(i,j), u_C, u_D, u_E, sigma_C, sigma_D, sigma_E, N_C/(N_C+N_D+N_E),N_D/(N_C+N_D+N_E),N_E/(N_C+N_D+N_E));
    end
end

s = contour(X, Y, classes,'b');
cfn_matrix = [0, 0, 0;
              0, 0, 0;
              0, 0, 0];
for i=1:size(rand_C, 1)
    gt = 1;
    pred = case2_map(rand_C(i, 1), rand_C(i, 2), u_C, u_D, u_E, sigma_C, sigma_D, sigma_E, N_C/(N_C+N_D+N_E),N_D/(N_C+N_D+N_E),N_E/(N_C+N_D+N_E));
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_D, 1)
    gt = 2;
    pred = case2_map(rand_D(i, 1), rand_D(i, 2), u_C, u_D, u_E, sigma_C, sigma_D, sigma_E, N_C/(N_C+N_D+N_E),N_D/(N_C+N_D+N_E),N_E/(N_C+N_D+N_E));
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_E, 1)
    gt = 3;
    pred = case2_map(rand_E(i, 1), rand_E(i, 2), u_C, u_D, u_E, sigma_C, sigma_D, sigma_E, N_C/(N_C+N_D+N_E),N_D/(N_C+N_D+N_E),N_E/(N_C+N_D+N_E));
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on

%%
% NN Case 1
a = linspace(-5, 20,100);
b = linspace(0, 25,100);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case1_nn(X(i,j), Y(i,j), rand_A, rand_B);
    end
end
s = contour(X, Y, classes, 'r');

hold on

%%
% NN Case 2
a = linspace(100, 500,1000);
b = linspace(100, 400,1000);

[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case2_nn(X(i,j), Y(i,j), rand_C, rand_D, rand_E);
    end
end

s = contour(X, Y, classes, 'r');

hold on

%%
% KNN Case 1
a = linspace(-5, 20,100);
b = linspace(0, 25,100);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case1_knn(X(i,j), Y(i,j), rand_A, rand_B);
    end
end

s = contour(X, Y, classes, 'g');

hold on

%%
% KNN Case 2
a = linspace(-10, 30,100);
b = linspace(-15, 30,100);
[X,Y] = meshgrid(a,b);

classes = zeros(numel(a), numel(b));

for i = 1:numel(a)
    for j = 1:numel(b)
        classes(i,j) = case2_knn(X(i,j), Y(i,j), rand_C, rand_D, rand_E);
    end
end

s = contour(X, Y, classes, 'g');

hold on
%%
% ERROR ANALYSIS

%%
% Generate Test set for NN and kNN

rand_A_test = randn(N_A,2)*chol(sigma_A)  + repmat(u_A,N_A,1);
rand_B_test = randn(N_B,2)*chol(sigma_B)  + repmat(u_B,N_B,1);
%%
% NN Case 1
cfn_matrix = [0, 0;
              0, 0];
for i=1:size(rand_A_test, 1)
    gt = 1;
    pred = case1_nn(rand_A_test(i, 1), rand_A_test(i, 2), rand_A, rand_B);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_B_test, 1)
    gt = 2;
    pred = case1_nn(rand_B_test(i, 1), rand_B_test(i, 2), rand_A, rand_B);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on
%%
% kNN Case 1
cfn_matrix = [0, 0;
              0, 0];
for i=1:size(rand_A_test, 1)
    gt = 1;
    pred = case1_knn(rand_A_test(i, 1), rand_A_test(i, 2), rand_A, rand_B);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_B_test, 1)
    gt = 2;
    pred = case1_knn(rand_B_test(i, 1), rand_B_test(i, 2), rand_A, rand_B);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on

%%

rand_C_test = randn(N_C,2)*chol(sigma_C)  + repmat(u_C,N_C,1);
rand_D_test = randn(N_D,2)*chol(sigma_D)  + repmat(u_D,N_D,1);
rand_E_test = randn(N_E,2)*chol(sigma_E)  + repmat(u_E,N_E,1);
%%
% NN Case 2
cfn_matrix = [0, 0, 0;
              0, 0, 0;
              0, 0, 0];
for i=1:size(rand_C_test, 1)
    gt = 1;
    pred = case2_nn(rand_C_test(i, 1), rand_C_test(i, 2), rand_C, rand_D, rand_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_D_test, 1)
    gt = 2;
    pred = case2_nn(rand_D_test(i, 1), rand_D_test(i, 2), rand_C, rand_D, rand_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
for i=1:size(rand_E_test, 1)
    gt = 3;
    pred = case2_nn(rand_E_test(i, 1), rand_E_test(i, 2), rand_C, rand_D, rand_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on
%%
% kNN Case 2
cfn_matrix = [0, 0, 0;
              0, 0, 0;
              0, 0, 0];
for i=1:size(rand_C_test, 1)
    gt = 1;
    pred = case2_knn(rand_C_test(i, 1), rand_C_test(i, 2), rand_C, rand_D, rand_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end

for i=1:size(rand_D_test, 1)
    gt = 2;
    pred = case2_knn(rand_D_test(i, 1), rand_D_test(i, 2), rand_C, rand_D, rand_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
for i=1:size(rand_E_test, 1)
    gt = 3;
    pred = case2_knn(rand_E_test(i, 1), rand_E_test(i, 2), rand_C, rand_D, rand_E);
    cfn_matrix(gt, pred) = cfn_matrix(gt, pred) + 1;
end
cfn_matrix
hold on