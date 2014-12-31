import sys
from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random

class Window(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        cbLayout = QtGui.QHBoxLayout()

        self.attitude_rotated = False
        cbAttitude = QtGui.QCheckBox('Attitude Rotated', self)
        cbAttitude.stateChanged.connect(self.toggleRotation)
        cbLayout.addWidget(cbAttitude)

        self.path_desired = False
        cbPathDesired = QtGui.QCheckBox('Path Desired', self)
        cbPathDesired.stateChanged.connect(self.togglePathDesired)
        cbLayout.addWidget(cbPathDesired)

        self.gps_position = True
        self.gps_velocity = True
        cbGPS = QtGui.QCheckBox('Show GPS', self)
        cbGPS.setCheckState(QtCore.Qt.Checked)
        cbGPS.stateChanged.connect(self.toggleGps)
        cbLayout.addWidget(cbGPS)

        self.stab_desired = False
        cbStabDesired = QtGui.QCheckBox('Stabilization Desired', self)
        cbStabDesired.stateChanged.connect(self.toggleStabDesired)
        cbLayout.addWidget(cbStabDesired)

        self.altitude = False
        cbAltitude = QtGui.QCheckBox('Altitude', self)
        cbAltitude.stateChanged.connect(self.toggleAltitude)
        cbLayout.addWidget(cbAltitude)

        # set the layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addLayout(cbLayout)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.uavo_list = None
        self.uavo_defs = None

    def resizeEvent( self, event ):
        if self.uavo_defs is not None:
            plt.tight_layout()

    def toggleRotation(self, state):
        self.attitude_rotated = state == QtCore.Qt.Checked
        self.draw()

    def togglePathDesired(self, state):
        self.path_desired = state == QtCore.Qt.Checked
        self.draw()

    def toggleGps(self, state):
        self.gps_position = state == QtCore.Qt.Checked
        self.gps_velocity = state == QtCore.Qt.Checked
        self.draw()

    def toggleStabDesired(self, state):
        self.stab_desired = state == QtCore.Qt.Checked
        self.draw()

    def toggleAltitude(self, state):
        self.altitude = state == QtCore.Qt.Checked
        self.draw()

    def plot(self, uavo_list, uavo_defs):
        ''' plot some random stuff '''

        self.uavo_list = uavo_list
        self.uavo_defs = uavo_defs

        self.draw()

    def draw(self):

        uavo_list = self.uavo_list
        uavo_defs = self.uavo_defs

        import taulabs
        uavo_classes = [(t[0], t[1]) for t in taulabs.uavo.__dict__.iteritems() if 'UAVO_' in t[0]]
        globals().update(uavo_classes)

        pos = uavo_list.as_numpy_array(UAVO_PositionActual)
        vel = uavo_list.as_numpy_array(UAVO_VelocityActual)
        att = uavo_list.as_numpy_array(UAVO_AttitudeActual)
        ned = uavo_list.as_numpy_array(UAVO_NEDPosition)
        pd = uavo_list.as_numpy_array(UAVO_PathDesired)

        plt.clf()

        # create an axis
        if self.altitude:
            ax1 = self.figure.add_subplot(231)
        else:
            ax1 = self.figure.add_subplot(221)
        ax1.plot(pos['time'],pos['North'],'.-',label="North")
        ax1.plot(pos['time'],pos['East'],'.-',label="East")
        if self.path_desired:
            ax1.plot(pd['time'],pd['End'][:,0], 'o-', label="PD North")
            ax1.plot(pd['time'],pd['End'][:,1], 'o-', label="PD East")
        if self.gps_position:
            ax1.plot(ned['time'],ned['North'], '*-', label="GPS North")
            ax1.plot(ned['time'],ned['East'], '*-', label="GPS East")
        plt.xlabel('Time (s)')
        plt.ylabel('Pos (m)')
        plt.legend()

        plt.xlim(pos['time'][0],pos['time'][0]+60)

        if self.altitude:
            ax2 = self.figure.add_subplot(232, sharex=ax1)
        else:
            ax2 = self.figure.add_subplot(222, sharex=ax1)
        ax2.plot(vel['time'],vel['North'],'.-',label="North")
        ax2.plot(vel['time'],vel['East'],'.-',label="East")
        if self.gps_velocity:
            gv = uavo_list.as_numpy_array(UAVO_GPSVelocity)
            ax2.plot(gv['time'],gv['North'],'*-',label="GPS North")
            ax2.plot(gv['time'],gv['East'],'*-',label="GPS East")
        plt.xlabel('Time (s)')
        plt.ylabel('Vel (m/s)')
        plt.legend()
        plt.ylim(-2,2)

        from numpy import cos,sin

        if self.altitude:
            ax3 = self.figure.add_subplot(234, sharex=ax1)
        else:
            ax3 = self.figure.add_subplot(223, sharex=ax1)
        if self.attitude_rotated:
            roll = att['Roll'][:,0]
            pitch = att['Pitch'][:,0]
            yaw = att['Yaw'][:,0]
            north = pitch * cos(yaw) + roll * -sin(yaw)
            east = pitch * sin(yaw) + roll * cos(yaw)
            ax3.plot(att['time'],north,'.-',label="North")
            ax3.plot(att['time'],east,'.-',label="East")
        else:
            ax3.plot(att['time'],att['Roll'],'.-',label="Roll")
            ax3.plot(att['time'],att['Pitch'],'.-',label="Pitch")

        if self.stab_desired:
            sd = uavo_list.as_numpy_array(UAVO_StabilizationDesired)
            ax3.plot(sd['time'],sd['Roll'],'o-',label="Roll Desired")
            ax3.plot(sd['time'],sd['Pitch'],'o-',label="Pitch Desired")
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.legend()
        plt.ylim(-20,20)

        if self.altitude:
            ax4 = self.figure.add_subplot(235, sharex=ax1)
        else:
            ax4 = self.figure.add_subplot(224, sharex=ax1)
        gyros = uavo_list.as_numpy_array(UAVO_Gyros)
        ax4.plot(gyros['time'],gyros['x'],'.-',label="Roll")
        ax4.plot(gyros['time'],gyros['y'],'.-',label="Pitch")
        ax4.plot(gyros['time'],gyros['z'],'.-',label="Yaw")
        plt.xlabel('Time (s)')
        plt.ylabel('Rate (deg/s)')
        plt.legend()

        if self.altitude:
            ax5 = self.figure.add_subplot(233, sharex=ax1)
            baro = uavo_list.as_numpy_array(UAVO_BaroAltitude)
            ax5.plot(pos['time'],-pos['Down'],label="Altitude")
            ax5.plot(baro['time'],baro['Altitude']-baro['Altitude'][0,0],label="Baro")
            if self.gps_position:
                ax5.plot(ned['time'],-ned['Down'],label="GPS")
            if self.path_desired:
                ax5.plot(pd['time'],-pd['End'][:,2], 'o-', label="PD Down")
            plt.xlabel('Time (s)')
            plt.ylabel('Pos (m)')
            plt.legend()

            ax6 = self.figure.add_subplot(236, sharex=ax1)
            ax6.plot(vel['time'],-vel['Down'],label="Estimate")
            if self.gps_velocity:
                gv = uavo_list.as_numpy_array(UAVO_GPSVelocity)
                ax6.plot(gv['time'],-gv['Down'],label="GPS")
            plt.xlabel('Time (s)')
            plt.ylabel('Vel (m/s)')
            plt.legend()
            plt.ylim(-2,2)

        plt.tight_layout()

        # refresh canvas
        self.canvas.draw()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
