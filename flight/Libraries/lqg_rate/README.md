
This is an LQG controller for rate control designed to complement the
autotuning algorithm. Specifically, it drops the parameter estimation
from the existing system identification kalman filter. This is designed
to run constantly and monitor the inputs to the motors in order to
estimate the instantaneous roll rate and motor torques, rather than
rely on just the gyros and noisy derivatives.

This kalman filter gives an estimate of the complete state of the rate
and torque which is then passed to an LQR controller. The parameters for
the LQR controller can be computed offline based on the autotune
parameters, at which point the control law becomes a simple matrix
multiplication on the state estimate and desired rates.

This code is licensed under the GPLv3, the full text of which can be
read here: can be read here: http://www.gnu.org/licenses/gpl-3.0.txt
