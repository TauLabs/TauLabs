
Ts = 1/400; % the time step

% discrete time formulation of dynamical model
% x[n+1] = A*x[n] + B*u

% set up the dynamical model
% x = [ax ay az wx wy wz tx wy tz]';  % angles, rates and torques
% u = [ux uy uz]';           % desired (prescaled) torques

% Discrete time dynamics

% rotation rate is slowly altered by torques and
% torques stay the same without input
b1 = exp(10.15)*Ts;
b2 = exp(9.63)*Ts;
b3 = exp(2.16)*Ts;
b3d = exp(7.69)*Ts;
t1 = Ts/(exp(-2.92) + Ts);

A = eye(9);
A(1:3,4:6) = Ts * eye(3);
A(4:6,7:9) = diag([b1 b2 b3]);
B = [zeros(3,3); diag([0 0 b3d]); t1*eye(3)];


% create cost matrices for LQR calculator. Note that we are using
% 12 states here as it is an augmented state with an integral error

q_angle = 500;
q_rate = 1;
q_torque = 1;
three = [1 1 1];
Q = diag([three*q_angle three*q_rate three*q_torque]);  % const on state errors
R = diag([1e4 1e4 1e5]);     % const on inputs
N = zeros(9,3);            % cross coupling costs between error and control

% Calculate LQR control weights if we have full state knowledge
% (which the corresponding kalman filter will provide)

sys = ss(A,B,diag(ones(1,9)),[],Ts,...
    'InputName',{'uR', 'uP', 'uY'}, ...
    'StateName',{'aR','aP','aY','wR','wP','wY','tR','tP','tY'}, ...
    'OutputName',{'aR','aP','aY','wR','wP','wY','tR','tP','tY'});
[L,S] = lqr(sys,Q,R,N);

L

% format matrix for C code
s = []; for i = 1:3; s1 = sprintf('%ff,',L(i,:)); s = [s '{' s1(1:end-1) sprintf('},\n')]; end; s

%L = dlqr(A,B,Q,R,N)