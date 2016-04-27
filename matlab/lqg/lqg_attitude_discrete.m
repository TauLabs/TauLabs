
Ts = 1/400; % the time step

% discrete time formulation of dynamical model
% x[n+1] = A*x[n] + B*u

% set up the dynamical model
% x = [a w t]';       % angles, rates and torques
% u = [u]';           % desired (prescaled) torques

% Discrete time dynamics

% rotation rate is slowly altered by torques and
% torques stay the same without input
b = exp(10.15)*Ts;
bd = exp(-10)*Ts;
t = Ts/(exp(-2.92) + Ts);

A = [1 Ts 0; 0 1 b; 0 0 1];
B = [0; bd; t];

% create cost matrices for LQR calculator.

q_angle = 1/(1)^2;
q_rate = 1/(100)^2;
q_torque = 1/(1e-2)^2;
Q = diag([q_angle q_rate q_torque]);  % const on state errors
R = 1;                                % const on inputs
N = zeros(3,1);                       % cross coupling costs between error and control

% Calculate LQR control weights if we have full state knowledge
% (which the corresponding kalman filter will provide)

sys = ss(A,B,diag(ones(1,3)),[],Ts,...
    'InputName',{'u'}, ...
    'StateName',{'a','w','t'}, ...
    'OutputName',{'a','w','t'});
[L,S] = lqr(sys,Q,R,N);

L

% format matrix for C code
s1 = sprintf('%ff,',L); s = ['{' s1(1:end-1) sprintf('},\n')]; s

%L = dlqr(A,B,Q,R,N)

%% can use same dynamics to calculate the controller for rate
% here the integral error should be 0 (instead of setpoint as
% attitude)

q_rate = 1/(10)^2;
q_torque = 1/(100)^2;
q_integral = 1e-10;

Q = diag([q_integral q_rate q_torque]);  % const on state errors
R = 1;     % const on inputs

[Lr,S] = lqr(sys,Q,R,N);
%s1 = sprintf('%ff,',Lr); s = ['{' s1(1:end-1) sprintf('},\n')]; s
Lr
