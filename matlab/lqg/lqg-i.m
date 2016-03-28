
Ts = 1/400; % the time step

% discrete time formulation of dynamical model
% x[n+1] = A*x[n] + B*u

% set up the dynamical model
% x = [wx wy wz tx wy tz]';  % rotation rates and torques
% u = [ux uy uz]';           % desired (prescaled) torques

% Discrete time dynamics

% rotation rate is slowly altered by torques and
% torques stay the same without input
b1 = exp(9.46)*Ts;
b2 = exp(9.21)*Ts;
b3 = exp(7.52)*Ts;
t1 = Ts/(exp(-3.3) + Ts);
t2 = Ts/(exp(-7.3) + Ts);

A = [1 0 0 b1 0  0  0 0 0;
     0 1 0 0  b2 0  0 0 0;
     0 0 1 0  0  b3 0 0 0;
     0 0 0 1  0  0  0 0 0;
     0 0 0 0  1  0  0 0 0;
     0 0 0 0  0  1  0 0 0;
     Ts 0 0 0 0  0  1 0 0;
     0 Ts 0 0 0  0  0 1 0;
     0 0 Ts 0 0  0  0 0 1];
B = [0 0 0;
     0 0 0;
     0 0 0;
     t1 0 0;
     0 t1 0;
     0 0 t2;
     0 0 0;
     0 0 0;
     0 0 0];
C = [1 0 0 0 0 0 0 0 0;
     0 1 0 0 0 0 0 0 0;
     0 0 1 0 0 0 0 0 0];
D = zeros(3,3);

% create cost matrices for LQR calculator. Note that we are using
% 12 states here as it is an augmented state with an integral error

q_rate = 10;
q_torque = 1;
q_integral = [10000 10000 1000];
three = [1 1 1];
Q = diag([three*q_rate three*q_torque q_integral]);  % const on state errors
R = diag([1 1 1]*10000);     % const on inputs
N = zeros(9,3);            % cross coupling costs between error and control

% Calculate LQR control weights if we have full state knowledge
% (which the corresponding kalman filter will provide)

sys = ss(A,B,diag(ones(1,9)),[],Ts,...
    'InputName',{'uR', 'uP', 'uY'}, ...
    'StateName',{'wR','wP','wY','tR','tP','tY','iwR','iwP','iwY'}, ...
    'OutputName',{'wR','wP','wY','tR','tP','tY','iwR','iwP','iwY'});
L = lqr(sys,Q,R,N);

L

s = []; for i = 1:3; s1 = sprintf('%ff,',L(i,:)); s = [s '{' s1(1:end-1) sprintf('},\n')]; end; s

%L = dlqr(A,B,Q,R,N)

%% Construct state estimator

Ak = [1 0 0 b1 0  0 ;
     0 1 0 0  b2 0  ;
     0 0 1 0  0  b3 ;
     0 0 0 1  0  0  ;
     0 0 0 0  1  0  ;
     0 0 0 0  0  1  ];
Bk = [0 0 0;
     0 0 0;
     0 0 0;
     t1 0 0;
     0 t1 0;
     0 0 t2];
Ck = [1 0 0 0 0 0;
     0 1 0 0 0 0 ;
     0 0 1 0 0 0 ];

G = [zeros(3,3); diag([1 1 1])*0.001]; % process noise on states
H = [zeros(3,3)];                      % noise on gyros

% create a model of system with process noise on the
% torque to account for model inaccuracies
sys_pn = ss(Ak,[Bk G],Ck,[D H],Ts, ...
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

