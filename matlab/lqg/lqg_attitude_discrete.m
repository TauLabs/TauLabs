
Ts = 1/400; % the time step

% discrete time formulation of dynamical model
% x[n+1] = A*x[n] + B*u

% set up the dynamical model
% x = [a w t]';       % angles, rates and torques
% u = [u]';           % desired (prescaled) torques

% Discrete time dynamics

% rotation rate is slowly altered by torques and
% torques stay the same without input
b = exp(10)*Ts;
bd = exp(-10)*Ts;
t = Ts/(exp(-3.5) + Ts);

A = [1 Ts 0; 0 1 b; 0 0 1];
B = [0; bd; t];

% create cost matrices for LQR calculator.

q_angle = 1/(1)^2;
q_rate = 1/(100)^2;
q_torque = 1/(100)^2*(b/Ts)^2;
Q = diag([q_angle q_rate q_torque]);  % const on state errors
R = 1;                                % const on inputs
N = zeros(3,1);                       % cross coupling costs between error and control

% solution is invariant on these transformations as long as we
% account on the output gains
k1 = 1e0;
A(2,3) = A(2,3) * k1;
B(3) = B(3) / k1;
R = R / (k1^2);
Q(3,3) = Q(3,3) * k1^2;

k2 = 1e0;
A(1,2) = A(1,2) * k2;
A(2,3) = A(2,3) / k2;
Q(2,2) = Q(2,2) * k2^2;

k3 = 1e0;
B(3) = B(3) / k3;
R = R / (k3 * k3);

k4 = 1e0;
Q = Q * k4;
R = R * k4;

% Calculate LQR control weights if we have full state knowledge
% (which the corresponding kalman filter will provide)

sys = ss(A,B,diag(ones(1,3)),[],Ts,...
    'InputName',{'u'}, ...
    'StateName',{'a','w','t'}, ...
    'OutputName',{'a','w','t'});
[L,S] = lqr(sys,Q,R,N);

conds = [cond(S) cond(inv(S)+B*inv(R)*B')] / 1e6;

disp(['L = ' sprintf('%f ', L)]);
disp(['conds = ' sprintf('%f ', conds)]);

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
