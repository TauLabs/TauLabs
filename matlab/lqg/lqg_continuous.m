
% continuous time formulation of dynamical model
% dx = A*x[n] + B*u

% set up the dynamical model
% x = [wx wy wz tx wy tz]';  % rotation rates and torques
% u = [ux uy uz]';           % desired (prescaled) torques

% Discrete time dynamics

% rotation rate is slowly altered by torques and
% torques stay the same without input
b1 = exp(9.46);
b2 = exp(9.21);
b3 = exp(7.52);
t1 = 1/exp(-3.3);
t2 = 1/exp(-7.3);
A = [zeros(3) diag([b1 b2 b3]);
     zeros(3,6)];
B = [zeros(3); 
     diag([t1 t1 t2])];
C = [eye(3) zeros(3)];
D = zeros(3);

% Calculate LQR control weights if we have full state knowledge
% (which the corresponding kalman filter will provide)

sys = ss(A,B,C,D,...
    'InputName',{'uR', 'uP', 'uY'}, ...
    'StateName',{'wR','wP','wY','tR','tP','tY'}, ...
    'OutputName',{'gR','gP','gY'});

q_rate = 10;
q_torque = 1;
q_integral = 0.01*[10000 10000 1000];
three = [1 1 1];
Q = diag([three*q_rate three*q_torque q_integral]);  % const on state errors
R = diag([1 1 1]*5000);        % const on inputs
N = zeros(6,3);           % cross coupling costs between error and control

Li = lqi(sys, Q, R)

%% Construct state estimator

G = [zeros(3,3); diag([1 1 1])]; % process noise on states
H = [zeros(3,3)];                      % process noise on gyros

% create a model of system with process noise on the
% torque to account for model inaccuracies
sys_pn = ss(A,[B G],C,[D H], ...
    'InputName',{'uR', 'uP', 'uY', 'vtR', 'vtP', 'vtY'}, ...
    'StateName',{'wR','wP','wY','tR','tP','tY'}, ...
    'OutputName',{'gR','gP','gY'});
%sys = ss(A,B,C,D,Ts)
% uX  -- control signals
% vtX -- torque drift
% wX  -- real rotation rate
% tX  -- real torque
% gX  -- gyro measurement

% create state estimator
Qn = diag([1 1 1]);    % magnitude of process noise
Rn = diag([1 1 1]);    % measurement noise

% from the dimensions of Qn and Rn kalman will interpret
% that the first three sys_pn.InputName are known inputs 
% and the second three are the process noise (in this case
% drift in the torque, without any additional input)
kest = kalman(sys_pn,Qn,Rn); %,[],[1 2 3],[1 2 3]);

%% Create the LQG controller

regulator = lqgtrack(kest, Li, '2dof');
controlled_sys = series(regulator,sys)
clsys = feedback(controlled_sys,eye(3),[4 5 6],[1 2 3],+1)

% version to allow inspecting what happens internally
claug = augstate(clsys);
claug = series(eye(3), claug, [1 2 3], [1 2 3])
%% Compare open and closed loop response to a control difference step

step(claug,0.15)
shg

