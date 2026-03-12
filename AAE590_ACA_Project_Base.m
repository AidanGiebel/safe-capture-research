%% AAE 590 Final Project Soft Capture Problem Umax code
clear;clc; close all;

% Boundary Condtitions
r0 = [20; 20; 10]; % Initial position (m)
v0 = [-0.02; -0.02; -0.01]; % Initial velocity (m/s)
x0 = [r0; v0];
rf = [0; 0; 0];  %Final position (m)
vf = [0; 0; 0];  %Final velocity (m/s)
xf = [rf; vf];

% Time and Vars
n = sqrt(3.986e5 / (6728^3));  %Mean motion (rad/s), ISS orbit
tf = 1000;  %Final time (sec)
N = 50; % Discretization node number
dt = tf/N;
t_k = 0; 
u0 = [0;0;0];

%Thrust and Vel Constraint
umax = 1e-4;
vmax = 0.031;

%Calculate Ak and Bk
A = [zeros(3), eye(3);
     3*n^2, 0, 0, 0, 2*n, 0;
     0, 0, 0, -2*n, 0, 0;
     0, 0, -n^2, 0, 0, 0];
B = [zeros(3,3); eye(3,3)];
c=[0;0;0;0;0;0];
[Ak, Bk] = compute_discrete_matrices(x0, u0, t_k, dt, A, B, c);


cvx_clear
cvx_begin
    variables x(6,N) u(3,N-1) 
    minimize sum(norms(u,2,1)) * dt
    subject to
    x(:,1) == x0;
    x(:,N) == xf;
    for k = 1:N-1
        x(:,k+1) == Ak * x(:,k) + Bk * u(:,k);
        % norm(u(:,k)) <= umax;
        % norm(x(4:6,k)) <= vmax;
    end
cvx_end

time = linspace(0, tf, N-1);

figure;
subplot(3,1,1);
plot(time, u(1,:), 'r', 'LineWidth', 2); hold on; grid
plot(time, u(2,:), 'g', 'LineWidth', 2);
plot(time, u(3,:), 'b', 'LineWidth', 2);
plot(time, vecnorm(u,2,1), '--', 'LineWidth', 2)
xlabel('Time (s)'); ylabel('Control (m/s^2)');
title('Optimal Control Inputs'); legend('u_x', 'u_y', 'u_z', 'norm u');

subplot(3,1,2);
plot(time, x(1,1:end-1), 'r', 'LineWidth', 2); hold on; grid
plot(time, x(2,1:end-1), 'g', 'LineWidth', 2);
plot(time, x(3,1:end-1), 'b', 'LineWidth', 2);
plot(time, vecnorm(x(1:3,1:end-1),2,1), '--', 'LineWidth', 2)
xlabel('Time (s)'); ylabel('Position (m)');
title('Position Trajectory'); legend('x', 'y', 'z', 'norm r');

subplot(3,1,3);
plot(time, x(4,1:end-1), 'r', 'LineWidth', 2); hold on; grid
plot(time, x(5,1:end-1), 'g', 'LineWidth', 2);
plot(time, x(6,1:end-1), 'b', 'LineWidth', 2);
plot(time, vecnorm(x(4:6,1:end-1),2,1), '--', 'LineWidth', 2)
xlabel('Time (s)'); ylabel('Velocity (m/s)');
title('Velocity Trajectory'); legend('v_x', 'v_y', 'v_z', 'norm v');

figure
plot3(x(1,:), x(2,:), x(3,:), 'b', 'LineWidth', 2); hold on;
plot3(x(1,1), x(2,1), x(3,1), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g'); %Start point
plot3(x(1,end), x(2,end), x(3,end), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); %End point
grid on; xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Relative Position Trajectory'); legend('Trajectory', 'Start', 'End');
axis equal;

function [A_k, B_k, c_k] = compute_discrete_matrices(x_k, u_k, t_k, dt, A, B, c)
    nx = length(x_k);
    nu = length(u_k);
    
    Y_k = [x_k; reshape(eye(nx), [], 1); zeros(nx * nu + nx, 1)];
    
    [~, Y_sol] = ode45(@(t, Y) EoMwithDiscreteMatrix(t, Y, u_k, nx, nu, A, B, c), [t_k, t_k + dt], Y_k);
    
    Y_k1 = Y_sol(end, :)';
    A_k = reshape(Y_k1(nx + 1:nx + nx^2), nx, nx);
    B_k = A_k * reshape(Y_k1(nx + nx^2 + 1:nx + nx^2 + nx * nu), nx, nu);
    c_k = A_k * Y_k1(nx + nx^2 + nx * nu + 1:nx +nx^2 + nx*nu + nx);
end

%% Function implementing Algorithm 4
function dYdt = EoMwithDiscreteMatrix(t, Y, u_k, nx, nu, A, B, c)
    x = Y(1:nx);
    Phi = reshape(Y(nx + 1:nx + nx^2), nx, nx);    
    dYdt = [
        A * x + B * u_k + c;
        reshape(A * Phi, [], 1);
        reshape(Phi \ B, [], 1);
        reshape(Phi \ c, [], 1)
    ];
end