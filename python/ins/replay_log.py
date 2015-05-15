#!/usr/bin/python -B

# Insert the parent directory into the module import search path.
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))

import argparse
import errno

from cins import CINS
import unittest

from sympy import symbols, lambdify, sqrt
from sympy import MatrixSymbol, Matrix
from numpy import cos, sin, power
from sympy.matrices import *
from quaternions import *
import numpy
import math
import ins

class ReplayFlightFunctions():

    def __init__(self):
        self.uavolist = []

    def load(self, filename):
        self.sim = CINS()
        self.sim.prepare()

        # add python directory to search path so the modules can be loaded
        import sys,os
        sys.path.insert(1, os.path.dirname(sys.path[0]))

        import taulabs
        import cPickle

        # load data from file
        handle = open(filename, 'rb')
        githash = cPickle.load(handle)
        print "Attempting to load data with githash of " + `githash`
        uavo_parsed = cPickle.load(handle)
        handle.close()

        # prepare the parser
        uavo_defs = taulabs.uavo_collection.UAVOCollection()
        uavo_defs.from_git_hash(githash)
        parser = taulabs.uavtalk.UavTalk(uavo_defs)

        print "Converting log records into python objects"
        uavo_list = taulabs.uavo_list.UAVOList(uavo_defs)
        for obj_id, data, timestamp in uavo_parsed:
            if obj_id in uavo_defs:
                obj = uavo_defs[obj_id]
                u = obj.instance_from_bytes(data, timestamp)
                uavo_list.append(u)
            else:
                print "Missing key: " + `obj_id`

        # We're done with this (potentially very large) variable, delete it.
        del uavo_parsed

        self.uavo_list = uavo_list

    def run_uavo_list(self):

        import taulabs
        import math

        uavo_list = self.uavo_list

        print "Replaying log file"

        attitude = uavo_list.as_numpy_array(taulabs.uavo.UAVO_AttitudeActual)
        gyros = uavo_list.as_numpy_array(taulabs.uavo.UAVO_Gyros)
        accels = uavo_list.as_numpy_array(taulabs.uavo.UAVO_Accels)
        #ned = uavo_list.as_numpy_array(taulabs.uavo.UAVO_NEDPosition)
        gps = uavo_list.as_numpy_array(taulabs.uavo.UAVO_GPSPosition)
        vel = uavo_list.as_numpy_array(taulabs.uavo.UAVO_GPSVelocity)
        mag = uavo_list.as_numpy_array(taulabs.uavo.UAVO_Magnetometer)
        baro = uavo_list.as_numpy_array(taulabs.uavo.UAVO_BaroAltitude)

        if gps.size == 0:
            print "Unable to process flight. No GPS data"
            return

        # set home location as first sample and linearize around that
        lat0 = gps['Latitude'][0,0]
        lon0 = gps['Longitude'][0,0]
        alt0 = gps['Altitude'][0,0]

        T = [alt0+6.378137E6, cos(lat0 / 10e6 * math.pi / 180.0)*(alt0+6.378137E6)]

        STEPS = gyros['time'].size
        history = numpy.zeros((STEPS,16))
        history_rpy = numpy.zeros((STEPS,3))
        times = numpy.zeros((STEPS,1))

        ned_history = numpy.zeros((gps['Latitude'].size,3))

        steps = 0
        t = gyros['time'][0]
        gyro_idx = 0
        accel_idx = 0
        gps_idx = 0
        vel_idx = 0
        mag_idx = 0
        baro_idx = 0

        dT = numpy.mean(numpy.diff(gyros['time']))

        for gyro_idx in numpy.arange(STEPS):

            t = gyros['time'][gyro_idx]
            steps = gyro_idx
            accel_idx = (numpy.abs(accels['time']-t)).argmin()
                
            gyros_dat = numpy.array([gyros['x'][gyro_idx],gyros['y'][gyro_idx],gyros['z'][gyro_idx]]).T[0]
            accels_dat = numpy.array([accels['x'][accel_idx],accels['y'][accel_idx],accels['z'][accel_idx]]).T[0]

            gyros_dat = gyros_dat / 180 * math.pi
            self.sim.predict(gyros_dat,accels_dat,dT)

            if (gps_idx < gps['time'].size) and (gps['time'][gps_idx] < t):
                pos = [(gps['Latitude'][gps_idx,0] - lat0) / 10e6 * math.pi / 180.0 * T[0], (gps['Longitude'][gps_idx,0] - lon0)  / 10e6 * math.pi / 180.0 * T[1], -(gps['Altitude'][gps_idx,0]-alt0)]
                self.sim.correction(pos=pos)
                ned_history[gps_idx,0] = pos[0]
                ned_history[gps_idx,1] = pos[1]
                ned_history[gps_idx,2] = pos[2]
                gps_idx = gps_idx + 1

            if (vel_idx < vel['time'].size) and (vel['time'][vel_idx] < t):
                self.sim.correction(vel=[vel['North'][vel_idx,0], vel['East'][vel_idx,0], vel['Down'][vel_idx,0]])
                vel_idx = vel_idx + 1

            if (mag_idx < mag['time'].size) and (mag['time'][mag_idx] < t):
                self.sim.correction(mag=[mag['x'][mag_idx,0], mag['y'][mag_idx,0], mag['z'][mag_idx,0]])
                mag_idx = mag_idx + 1

            if (baro_idx < baro['time'].size) and (baro['time'][baro_idx] < t):
                self.sim.correction(baro=baro['Altitude'][baro_idx,0])
                baro_idx = baro_idx + 1

            history[steps,:] = self.sim.state
            history_rpy[steps,:] = quat_rpy(self.sim.state[6:10])
            times[steps] = t
        print "Plotting results"
       
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,2,sharex=True)
        ax[0][0].cla()
        ax[0][0].plot(gps['time'],ned_history[:,0],'b--',label="North")
        ax[0][0].plot(gps['time'],ned_history[:,1],'g--',label="East")
        ax[0][0].plot(gps['time'],ned_history[:,2],'r--',label="Down"),
        ax[0][0].plot(baro['time'], -baro['Altitude'],'--',label="Baro")
        ax[0][0].plot(times,history[:,0],'b',label="Replay")
        ax[0][0].plot(times,history[:,1],'g',label="Replay")
        ax[0][0].plot(times,history[:,2],'r',label="Replay")
        ax[0][0].set_title('Position')
        plt.sca(ax[0][0])
        plt.ylabel('m')
        plt.legend()

        ax[0][1].cla()
        ax[0][1].plot(vel['time'],vel['North'],'b--',label="North")
        ax[0][1].plot(vel['time'],vel['East'],'g--',label="East")
        ax[0][1].plot(vel['time'],vel['Down'],'r--',label="Down")
        ax[0][1].plot(times,history[:,3],'b',label="Replay")
        ax[0][1].plot(times,history[:,4],'g',label="Replay")
        ax[0][1].plot(times,history[:,5],'r',label="Replay")
        ax[0][1].set_title('Velocity')
        plt.sca(ax[0][1])
        plt.ylabel('m/s')
        plt.legend()

        ax[1][0].cla()
        ax[1][0].plot(attitude['time'],attitude['Roll'],'--',label="Roll")
        ax[1][0].plot(attitude['time'],attitude['Pitch'],'--',label="Pitch")
        ax[1][0].plot(attitude['time'],attitude['Yaw'],'--',label="Yaw")
        ax[1][0].plot(times,history_rpy[:,:], label="Replay")
        ax[1][0].set_title('Attitude')
        plt.sca(ax[1][0])
        plt.ylabel('Angle (Deg)')
        plt.xlabel('Time (s)')
        plt.legend()

        ax[1][1].cla()
        ax[1][1].plot(times,history[:,10:13]*180/3.1415,label="Gyro Bias")
        ax[1][1].plot(times,history[:,-1],label="Z Bias")
        ax[1][1].set_title('Biases')
        plt.sca(ax[1][1])
        plt.ylabel('Bias (rad/s)')
        plt.xlabel('Time (s)')
        plt.legend()

        plt.draw()
        plt.show()

        return ins

#-------------------------------------------------------------------------------
USAGE = "%(prog)s [logfile.pickle]"
DESC  = """
  Load a Tau Labs pickled log file and reruns it through the INS. Requires a high
  resolution log with almost all the data.
"""

if __name__ == '__main__':
    # Setup the command line arguments.
    parser = argparse.ArgumentParser(usage = USAGE, description = DESC)

    parser.add_argument("logfile",
                        nargs = "+",
                        help  = "list of log files for processing")

    # Parse the command-line.
    args = parser.parse_args()

    replay = ReplayFlightFunctions()
    replay.load(args.logfile[0])
    replay.run_uavo_list()