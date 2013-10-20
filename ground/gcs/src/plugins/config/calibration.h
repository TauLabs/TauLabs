    /**
 ******************************************************************************
 *
 * @file       calibration.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Gui-less support class for calibration
 *****************************************************************************/
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <config_global.h>
#include <uavobjectmanager.h>
#include "uavobjectutil/uavobjectutilmanager.h"
#include <extensionsystem/pluginmanager.h>
#include <uavobject.h>
#include <tempcompcurve.h>

#include <QObject>
#include <QTimer>
#include <QString>

/**
 * @brief The Calibration class is a UI free algorithm that can be connected
 * to any interfaces.  As such it only communicates with the UI via signals
 * and slots, but has no direct handles to any particular controls or widgets.
 *
 * It performs a number of calibration routines, including six-point calibration
 * for accelerometers and magnetometers, temperature compensation for gyros,
 * calculating the rotation to level the accelerometers, and calculating the
 * board orientation.
 */
class CONFIG_EXPORT Calibration : public QObject
{
    Q_OBJECT

public:
    explicit Calibration();
    ~Calibration();

    void initialize(bool calibrateAccels, bool calibrateMags);

private:
    enum CALIBRATION_STATE {
        IDLE, LEVELING, YAW_ORIENTATION,
        SIX_POINT_WAIT1, SIX_POINT_COLLECT1,
        SIX_POINT_WAIT2, SIX_POINT_COLLECT2,
        SIX_POINT_WAIT3, SIX_POINT_COLLECT3,
        SIX_POINT_WAIT4, SIX_POINT_COLLECT4,
        SIX_POINT_WAIT5, SIX_POINT_COLLECT5,
        SIX_POINT_WAIT6, SIX_POINT_COLLECT6,
        GYRO_TEMP_CAL
    } calibration_state;

public slots:
    //! Start collecting data while vehicle is level
    void doStartBiasAndLeveling();

    //! Start collecting data while vehicle is level
    void doStartNoBiasLeveling();

    //! Start collecting data while vehicle is in pure pitch
    void doStartOrientation();

    //! Start the six point calibration routine
    void doStartSixPoint();

    //! Cancels the six point calibration routine
    void doCancelSixPoint();

    //! Indicates UAV is in a position to collect data during 6pt calibration
    void doSaveSixPointPosition();

    //! Start collecting gyro temp calibration data
    void doStartTempCal();

    //! Accept gyro temp calibration data
    void doAcceptTempCal();

    //! Cancels the temperature calibration routine
    void doCancelTempCalPoint();

    //! Set up the curves
    void configureTempCurves(TempCompCurve *x, TempCompCurve *y, TempCompCurve *z);

private slots:
    //! New data acquired
    void dataUpdated(UAVObject *);

    //! Data collection timed out
    void timeout();

public slots:
    //! Set temperature calibration range
    void setTempCalRange(int r);

signals:
    //! Indicate whether to enable or disable controls
    void toggleControls(bool enable);

    //! Indicate whether to enable or disable controls
    void toggleSavePosition(bool enable);

    //! Change the UAV visualization
    void updatePlane(int position);

    //! Show an instruction to the user for six point calibration
    void showSixPointMessage(QString message);

    //! Show an instruction to the user for yaw orientation
    void showYawOrientationMessage(QString message);

    //! Show an instruction to the user for leveling
    void showLevelingMessage(QString message);

    //! Indicate what the progress is for yaw orientation
    void yawOrientationProgressChanged(int);

    //! Indicate what the progress is for leveling
    void levelingProgressChanged(int);

    //! Indicate what the progress is for six point collection
    void sixPointProgressChanged(int);

    //! Show an instruction or message from temperature calibration
    void showTempCalMessage(QString message);

    //! Indicate what the progress is for temperature calibration
    void tempCalProgressChanged(int);

    //! Indicate that a calibration process has successfully completed and the results saved to UAVO
    void calibrationCompleted();

private:
    //! Perform the leveling calculation
    void doStartLeveling();

    //! Get the object manager
    UAVObjectManager* getObjectManager();

    //! Get the object manager utility
    UAVObjectUtilManager* getObjectUtilManager();

    //! Assign the metadata update rate
    void assignUpdateRate(UAVObject* obj, quint32 updatePeriod);

    QTimer timer;

    //! Whether to attempt to calibrate the magnetometer (normally if it is present)
    bool calibrateMags;

    //! Whether to attempt to calibrate the accelerometer (normally if it is present)
    bool calibrateAccels;

    //! Whether to zero the z-accel
    bool zeroVertical;

    //! The expected gravity amplitude
    double accelLength;

    //! Store the initial metadata in order to restore it after calibration
    QMap<QString, UAVObject::Metadata> originalMetaData;

    //! List of optimized metadata rates
    QMap<QString, UAVObject::Metadata> metaDataList;

    QList<double> gyro_accum_x;
    QList<double> gyro_accum_y;
    QList<double> gyro_accum_z;
    QList<double> gyro_accum_temp;
    QList<double> accel_accum_x;
    QList<double> accel_accum_y;
    QList<double> accel_accum_z;
    QList<double> mag_accum_x;
    QList<double> mag_accum_y;
    QList<double> mag_accum_z;

    double gyro_data_x[6], gyro_data_y[6], gyro_data_z[6];
    double accel_data_x[6], accel_data_y[6], accel_data_z[6];
    double mag_data_x[6], mag_data_y[6], mag_data_z[6];

    static const int NUM_SENSOR_UPDATES_LEVELING = 300;
    static const int NUM_SENSOR_UPDATES_YAW_ORIENTATION = 300;
    static const int NUM_SENSOR_UPDATES_SIX_POINT = 100;
    static const int SENSOR_UPDATE_PERIOD = 20;
    static const int NON_SENSOR_UPDATE_PERIOD = 5000;
    double MIN_TEMPERATURE_RANGE;

    double initialBoardRotation[3];
    double initialAccelsScale[3];
    double initialAccelsBias[3];
    double initialMagsScale[3];
    double initialMagsBias[3];

    TempCompCurve *xCurve;
    TempCompCurve *yCurve;
    TempCompCurve *zCurve;

protected:
    enum sensor_type {ACCEL, GYRO, MAG};

    //! Connect and speed up or disconnect a sensor
    void connectSensor(sensor_type sensor, bool connect);

    //! Store a measurement at this position and indicate if it is the last one
    bool storeSixPointMeasurement(UAVObject * obj, int position);

    //! Store yaw orientation sample and compute orientation if finished
    bool storeYawOrientationMeasurement(UAVObject *obj);

    //! Store leveling sample and compute level if finished
    bool storeLevelingMeasurement(UAVObject *obj);

    //! Computes the scale and bias for the accelerometer and mag
    int computeScaleBias();

    int SixPointInConstFieldCal( double ConstMag, double x[6], double y[6], double z[6], double S[3], double b[3] );
    int LinearEquationsSolving(int nDim, double* pfMatr, double* pfVect, double* pfSolution);

    //! Rotate a vector by the rotation matrix, optionally trasposing
    void rotate_vector(double R[3][3], const double vec[3], double vec_out[3], bool transpose);

    //! Compute a rotation matrix from a set of euler angles
    void Euler2R(double rpy[3], double Rbe[3][3]);

    //! Compute the mean value of a list
    static double listMean(QList<double> list);

    //! Compute the min value of a list
    static double listMin(QList<double> list);

    //! Compute the max value of a list
    static double listMax(QList<double> list);

    //! Reset sensor settings to pre-calibration values
    void resetSensorCalibrationToOriginalValues();

    //! Store a sample for temperature compensation
    bool storeTempCalMeasurement(UAVObject *obj);

    //! Compute temperature compensation factors
    int computeTempCal();

    //! Update the graphs with the temperature compensation
    void updateTempCompCalibrationDisplay();

};

#endif // CALIBRATION_H
