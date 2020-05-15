%% Тест 1. Квадратичный функционал, ограничение - эллипс
clc;
f = @(x) (5 * x(2) - 2)^2 +(2 * x(1) - 0.7)^2 - 2;
rho = supportLebesgue(f, optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp'), 2);

J = @(x, y) A(1, 1) * x .* x + A(1, 2) * x .* y + A(2, 1) * x .* y  + A(2, 2) * y .* y + c(1) * x + c(2) * y;
x = linspace(-1, 1.5, 100);
y = linspace(-1, 1, 100);
[X, Y] = meshgrid(x, y);
contour(X, Y, J(X, Y), 100);
hold on;

A = [7 2; 1 4];
c = [2; 1];
[~, ~] = cond_grad(A, c, rho, [0.5; 0.6], 0.000001);
[~, ~] = conj_grad(A, c, [1; -0.8], 0.000001);

drawSet(@(x) rho(x), 100);
axis equal;
hold on;


%% Тест 2. 4 степень
clc;
f = @(x) (x(1))^4 + (x(2) - 0.5)^2 - 0.05;
rho = supportLebesgue(f, optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp'), 2);

x = linspace(-0.8, 0.8, 100);
y = linspace(-0.8, 0.8, 100);
[X, Y] = meshgrid(x, y);
JJ = @(x, y) x.^4 + y.^4 + (x + y).^2;
contour(X, Y, JJ(X, Y), 200);
hold on;

J = @(u) u(1).^4 + u(2).^4 + (u(1) + u(2)).^2;
grad = @(u) [4 * u(1)^3 + 2 * (u(1) + u(2)); 4 * u(2) ^ 3 + 2 * (u(1) + u(2))];


[~, ~] = cond_grad([], [], rho, [0.1; 0.51], 0.0001, J, grad);

[u_min, J_min] = conj_grad([], [], [0.5; 0], 0.0001, J, grad);



plot(0, 0, 'x black');

drawSet(@(x) rho(x), 100);
axis equal;
hold on;


%% Тест 2. Функция Химмельблау
clc;
f = @(x) (x(1) -3)^4 + (x(2) - 2)^4 - 0.5;
rho = supportLebesgue(f, optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp'), 2);

x = linspace(-4, 4, 100);
y = linspace(-4, 4, 100);
[X, Y] = meshgrid(x, y);
JJ = @(x, y) (x.^2 + y - 11).^2 + (x + y.^2 - 7).^2;
contour(X, Y, JJ(X, Y), 30);
hold on;

J = @(u) (u(1).^2 + u(2) - 11).^2 + (u(1) + u(2).^2 - 7).^2;
grad = @(u) [4 * u(1) * (u(1).^2 + u(2) - 11) + 2 * (u(1) + u(2).^2 - 7);  (u(1).^2 + u(2) - 11)*2 + (u(1) + u(2).^2 - 7)*4*u(2)];


[~, ~] = cond_grad([], [], rho, [3.6; 2.7], 0.0001, J, grad);

[u_min, J_min] = conj_grad([], [], [2; -3], 0.0001, J, grad);




drawSet(@(x) rho(x), 100);
axis equal;
hold on;


%% Тест 2. n-мерный пример
clc;
f = @(x) sq_norm(x) - 1;
n = 10;
rho = supportLebesgue(f, optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp'), n);
A = diag(ones(1, n));
c = zeros(n, 1);
c(2) = 1;
u0 = zeros(n, 1);
u0(1) = 0.5;
u0(2) = 0.5;
u0(3) = 0.3;
[~, ~] = cond_grad(A, c, rho, u0, 0.00001);
[~, ~] = conj_grad(A, c, u0, 0.0001);

