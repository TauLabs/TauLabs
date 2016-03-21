
Ts = 1/400; % the time step

% discrete time formulation of dynamical model
% x[n+1] = A*x[n] + B*u

% set up the dynamical model
% x = [wx wy wz tx wy tz]';  % rotation rates and torques
% u = [ux uy uz]';           % desired (prescaled) torques

% Discrete time dynamics

% rotation rate is slowly altered by torques and
% torques stay the same without input
b = 0.01;
A = [1 0 0 b 0 0;
     0 1 0 0 b 0;
     0 0 1 0 0 b;
     0 0 0 1 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 1];
B = [0 0 0;
     0 0 0;
     0 0 0;
     Ts 0 0;
     0 Ts 0;
     0 0 Ts];
C = [1 0 0 0 0 0;
     0 1 0 0 0 0;
     0 0 1 0 0 0];
D = zeros(3,3);

% create cost matrices for LQR calculator
Q = diag([1 1 1 1 1 1]);  % const on state errors
R = diag([1 1 1]);        % const on inputs
N = zeros(6,3);           % cross coupling costs between error and control

% Calculate LQR control weights if we have full state knowledge
% (which the corresponding kalman filter will provide)

sys = ss(A,B,diag(ones(1,6)),[],Ts,...
    'InputName',{'uR', 'uP', 'uY'}, ...
    'StateName',{'wR','wP','wY','tR','tP','tY'}, ...
    'OutputName',{'wR','wP','wY','tR','tP','tY'});
L = lqr(sys,Q,R,N);

%L = dlqr(A,B,Q,R,N)

%% Construct state estimator

G = [zeros(3,3); diag([1 1 1])*0.001]; % process noise on states
H = [zeros(3,3)];                      % noise on gyros

% create a model of system with process noise on the
% torque to account for model inaccuracies
sys_pn = ss(A,[B G],C,[D H],Ts, ...
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

regulator = lqgreg(kest, L);
sys = ss(A,B,C,D,Ts,...
    'InputName',{'uR', 'uP', 'uY'}, ...
    'StateName',{'wR','wP','wY','tR','tP','tY'}, ...
    'OutputName',{'gR','gP','gY'});

clsys = feedback(sys, regulator, +1)

%% Compare open and closed loop response to a control difference step

step(clsys,sys*0.01)

