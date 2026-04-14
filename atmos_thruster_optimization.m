% ATMOS Co-design Optimization 

import casadi.*

%% Parameters
N = 50;         % number of time steps
n = 10;         % number of sampled trajectories
num_actuators = 16;
nx = 13;        % state dimension
nu = 16;        % actuator dimension
dt = 0.1;       % time step
alpha = 100.0;   % L1 penalty weight -> Good values: 100, 
m_kg = 16.8;        % mass
I = [
    [0.48425, 0.00037320, -0.010538],
    [0.00037320, 0.51704, 0.0051258],
    [-0.010538, 0.0051258, 0.28563]
]; % Moment of inertia

I_inv = inv(I);

u_max = 20.0; % Thruster force limit
% r_min = -0.1819;
r_max = 0.1819;

%% Dynamics Setup
x_sym = MX.sym('x', nx);
u_sym = MX.sym('u', nu);
r_sym = MX.sym('r', 3, num_actuators);
beta_sym = MX.sym('beta', num_actuators);

% Variable Mixer Matrix, changes with r (actuator placements)
d_base = [
    0, 0, 0, 0,   0,  0,  0, 0,  -1, 1, -1, 1,    0, 0, 0, 0,    0,  0,  0,  0,   0, 0, 0, 0; % Fx
   -1,-1, 1, 1,  -1, -1,  1, 1,   0, 0,  0, 0,    0, 0, 0, 0,    0,  0,  0,  0    0, 0, 0, 0; % Fy
    0, 0, 0, 0,   0,  0,  0, 0,   0, 0,  0, 0,    1, 1, 1, 1,   -1, -1, -1, -1    -1 -1 -1 -1; % Fz
];

B_sym = []; % Init mixer matrix B
for i = 1:num_actuators
    F_i = d_base(:, i); % Force direction (fixed)
    T_i = cross(r_sym(:, i), F_i); % Torque lever arm (variable)
    B_sym = horzcat(B_sym, [F_i; T_i]); % Concatenate columns
end

% Helper functions for rotation and kinematics
rot_m = @(q) [1-2*(q(3)^2+q(4)^2), 2*(q(2)*q(3)-q(1)*q(4)), 2*(q(2)*q(4)+q(1)*q(3));
              2*(q(2)*q(3)+q(1)*q(4)), 1-2*(q(2)^2+q(4)^2), 2*(q(3)*q(4)-q(1)*q(2));
              2*(q(2)*q(4)-q(1)*q(3)), 2*(q(3)*q(4)+q(1)*q(2)), 1-2*(q(2)^2+q(3)^2)];

kin_m = @(q) [-q(2), q(1), -q(4), q(3);
              -q(3), q(4), q(1), -q(2);
              -q(4), -q(3), q(2), q(1)];

forces_torques = B_sym * (u_sym); 
F_body = forces_torques(1:3);
T_body = forces_torques(4:6);

% Extract states 
q = x_sym(4:7); q_unit = q/sqrt(sumsqr(q) + 1e-6);
v = x_sym(8:10); omega = x_sym(11:13);

% derivative of states in body frame (x_dot)
p_dot = rot_m(q_unit) * v; 
v_dot = (1/m_kg) * F_body - cross(omega, v);
q_dot = 0.5 * kin_m(q_unit)' * omega; % quaternion from inertial to body frame
omega_dot = I_inv * (T_body - cross(omega, I * omega)); 


f_dynamics = Function('f', {x_sym, u_sym, beta_sym, r_sym}, {vertcat(p_dot, q_dot, v_dot, omega_dot)});

%% Optimization Problem
opti = Opti();

% Decision Variables
x_vars = cell(n, 1);
u_vars = cell(n, 1);
for i = 1:n
    x_vars{i} = opti.variable(nx, N);
    u_vars{i} = opti.variable(nu, N-1);
end
r = opti.variable(3, num_actuators); % Actuator positions variable
beta = opti.variable(num_actuators);

% Objective

% rotations
eul_x = [pi/2 0 0]; quat_x = eul2quat(eul_x);
eul_y = [0 pi/2 0]; quat_y = eul2quat(eul_y);
eul_z = [0 0 pi/2]; quat_z = eul2quat(eul_z);

start_state = [zeros(3,1); 1.0; zeros(9,1)]; % change to user input later
W1 = start_state;
W2 = [0.5; 1.0; 1.5; quat_x'; zeros(6,1)];
W3 = [0.8; 0.5; 2.0; quat_y'; zeros(6,1)];
W4 = [1.0; 2.0; 3.0; quat_z'; zeros(6,1)];
waypoints = [W1, W2, W3, W4];
num_waypoints = size(waypoints,2);

% Warm start with reference trajectory
steps_per_segment = floor(N/num_waypoints-1);
x_ref = zeros(13,N);

for i = 1:(num_waypoints-1)
    idx_start = (i-1)*steps_per_segment + 1;
    if i == num_waypoints - 1
        idx_end = N;
    else 
        idx_end = i*steps_per_segment;
    end

    segment_N = idx_end - idx_start + 1;

    for k = 0:(segment_N - 1)
        s = k/(segment_N - 1);
        a = s^2*(3-2*s);
        % linearly interpolate position
        x_ref(1:3,idx_start + k) = (1-a) * waypoints(1:3,i) + a*waypoints(1:3, i+1);

        % SLERP for rotations
        q_start = waypoints(4:7, i);
        q_end = waypoints(4:7, i+1);
        quat_start = quaternion(q_start');
        quat_end = quaternion(q_end');
        quat_interp = slerp(quat_start, quat_end, a);
        x_ref(4:7, idx_start+k) = compact(quat_interp)';
        
        x_ref(8:13, idx_start + k) = (1-a) * waypoints(8:13, i) + a*waypoints(8:13, i+1);
    end
end

% Weighting matrix
Q = diag([10, 10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1]);
R = 0.1*eye(num_actuators);
total_loss = 0;
for i = 1:n
% 1. State Tracking Loss: Sum over all N time steps
    for k = 1:N
        err = x_vars{i}(:, k) - x_ref(:, k); % Column k vs Column k
        total_loss = total_loss + err' * Q * err;
        
        % total_loss = total_loss + err' * Q * err + 10* (sumsqr(x_vars{i}(4:7, k)-1)^2);
    end
    for k = 1:N-1
        u_k = u_vars{i}(:,k);
        total_loss = total_loss + u_k' * R * u_k;
    end
end

dist_penalty = 0;
for i = 1:num_actuators
    for j = i+1:num_actuators
        dist_sq = sumsqr(r(:,i) - r(:,j));
        dist_penalty = dist_penalty + 1 / (dist_sq + 1e-4); 
    end
end
total_loss = total_loss + 0.01 * dist_penalty;

total_loss = total_loss + alpha * sum(sqrt(beta.^2+10^(-6)));
opti.minimize(total_loss);

%% Constraints

% Set initial state as the start_state + noise
sampled_x0 = repmat(start_state, 1, n) + randn(nx, n) * 0.001;

for i = 1:n
    quat_part = sampled_x0(4:7, i);
    sampled_x0(4:7, i) = quat_part / norm(quat_part); % Ensure norm is 1.0
end

% Unit quaternion constraint
opti.subject_to(sumsqr(x_vars{1}(4:7, 1))==1);
opti.subject_to(sumsqr(x_vars{n}(4:7, N))==1);

% IC: start all thrusters around the edges of the circular plate
theta = linspace(0, 2*pi, num_actuators + 1);
theta = theta(1:end-1); % remove redundant last point
% Calculate positions at the edge of r_max
r_guess = zeros(3, num_actuators);
r_guess(1, :) = r_max * cos(theta); % X coordinates
r_guess(2, :) = r_max * sin(theta); % Y coordinates
r_guess(3, :) = 0;                  % Z coordinates (or vary if 3D)

opti.set_initial(r, r_guess);

opti.subject_to(0 <= beta <= 1);
opti.subject_to(-r_max <= r <= r_max);
for i = 1:n
    opti.subject_to(0 <= u_vars{i} <= u_max); % control bounds
    opti.subject_to(x_vars{i}(:, 1) == sampled_x0(:, i)); % IC

    % midpoint integration
    for k = 1:N-1
        u_k = u_vars{i}(:, k);
        opti.subject_to(0 <= u_k <= u_max);
        x_mid = (x_vars{i}(:, k) + x_vars{i}(:, k+1)) / 2;
        opti.subject_to(x_vars{i}(:, k+1) == x_vars{i}(:, k) + dt * f_dynamics(x_mid, u_k, beta, r));
    end
end


%% Solver

% Solver settings
opts = struct;
opts.ipopt.tol = 1e-8;           % default 1e-8
opts.ipopt.dual_inf_tol = 1e-8;  % default 1e-8
opts.ipopt.constr_viol_tol = 1e-4;
opts.ipopt.max_iter = 300;

opti.solver('ipopt', opts);
% opti.callback(@(i) disp(['Iteration: ' num2str(i)]));
sol = opti.solve();

%% Solutions and Plots
final_beta = sol.value(beta);
optimized_configuration = (final_beta > 1e-8);
disp('Optimized Actuator Configuration:');
disp(double(optimized_configuration));

% Extract the optimized values
r_val = sol.value(r);
beta_val = sol.value(beta);

% Display r 
fprintf('--- Optimized Thruster Positions (r) ---\n');
disp(r_val);

% Display beta (your 16x1 activation vector)
fprintf('--- Optimized Thruster Activations (beta) ---\n');
disp(beta_val'); % Transposed for easier reading

% Import atmos model
fv = stlread('C:\Users\natha\Downloads\ATMOS - PreRelease_v0.2\ATMOS - Release\Alpha3_v4.STL'); %faces and vertices of model
pts = fv.Points; % vertices
faces = fv.ConnectivityList;

center_offset = mean(pts);
pts = pts - center_offset;

figure
patch('Faces', faces, 'Vertices', pts, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;

% Plot thruster positions

rx = r_val(1, :);
ry = r_val(2,: );
rz = r_val(3, :);

scatter3(rx, ry, rz, 200, 'filled', 'MarkerFaceColor','r', 'MarkerEdgeColor', 'k')
xlim([-0.5, 0.5]); ylim([-0.8, 0.8]); zlim([-0.8, 0.8]);
% 5. Aesthetics
axis equal;
grid on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
view(3); % Set to 3D view
camlight; lighting gouraud; % Makes the CAD look 3D