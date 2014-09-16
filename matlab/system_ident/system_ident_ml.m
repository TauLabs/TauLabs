function system_ident_ml
% [X, P] = system_ident_ml(X,P,Y,U)
% [X, P] = system_ident_ml()
%
% Calls the matlab wrapper for the system identification EKF written
% in C.
%
% X is the 13 element state matrix consisting of
%  - roll rate estimate
%  - pitch rate estimate
%  - yaw rate estimate
%  - scaled roll torque 
%  - scaled pitch torque
%  - scaled yaw torque
%  - roll torque scale
%  - pitch torque scale
%  - yaw torque scale
%  - time response of the motors
%  - bias in the roll torque
%  - bias in the pitch torque
%  - bias in the yaw torque
%
% P is the covariance array, but represented as a linear array of
% the non-zero elements.
%
% Y is the gyro elements in [roll, pitch, yaw] format
%
% U is the control input in [roll, pitch, yaw] format
%
% Call with no input to get initial conditionals for X and P