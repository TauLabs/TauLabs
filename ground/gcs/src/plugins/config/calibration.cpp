/**
 ******************************************************************************
 * @file       calibration.cpp
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
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

#include "calibration.h"

#include "utils/coordinateconversions.h"
#include <QMessageBox>
#include <QDebug>
#include <QThread>
#include "accels.h"
#include "gyros.h"
#include "magnetometer.h"
#include "revocalibration.h"
#include "homelocation.h"
#include "attitudesettings.h"
#include "inertialsensorsettings.h"


class Thread : public QThread
{
public:
    static void usleep(unsigned long usecs)
    {
        QThread::usleep(usecs);
    }
};

enum calibrationSuccessMessages{
    CALIBRATION_SUCCESS,
    ACCELEROMETER_FAILED,
    MAGNETOMETER_FAILED
};

#define sign(x) ((x < 0) ? -1 : 1)

Calibration::Calibration() : calibrateMag(false), accelLength(9.81)
{
}

Calibration::~Calibration()
{
}

/**
 * @brief Calibration::initialize Configure whether to calibrate the mag during 6 point cal
 * @param calibrateMags
 */
void Calibration::initialize(bool calibrateMags) {
    this->calibrateMag = calibrateMags;
}

/**
 * @brief Calibration::connectSensor
 * @param sensor The sensor to change
 * @param con Whether to connect or disconnect to this sensor
 */
void Calibration::connectSensor(sensor_type sensor, bool con)
{
    UAVObject::Metadata mdata;
    if (con) {
        switch (sensor) {
        case ACCEL:
        {
            Accels * accels = Accels::GetInstance(getObjectManager());
            Q_ASSERT(accels);

            initialAccelsMdata = accels->getMetadata();
            mdata = initialAccelsMdata;
            UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
            mdata.flightTelemetryUpdatePeriod = SENSOR_UPDATE_PERIOD;
            accels->setMetadata(mdata);

            connect(accels, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        case MAG:
        {
            Magnetometer * mag = Magnetometer::GetInstance(getObjectManager());
            Q_ASSERT(mag);

            initialMagMdata = mag->getMetadata();
            mdata = initialMagMdata;
            UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
            mdata.flightTelemetryUpdatePeriod = SENSOR_UPDATE_PERIOD;
            mag->setMetadata(mdata);

            connect(mag, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        case GYRO:
        {
            Gyros * gyros = Gyros::GetInstance(getObjectManager());
            Q_ASSERT(gyros);

            initialGyrosMdata = gyros->getMetadata();
            mdata = initialMagMdata;
            UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
            mdata.flightTelemetryUpdatePeriod = SENSOR_UPDATE_PERIOD;
            gyros->setMetadata(mdata);

            connect(gyros, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        }
    } else {
        switch (sensor) {
        case ACCEL:
        {
            Accels * accels = Accels::GetInstance(getObjectManager());
            Q_ASSERT(accels);
            accels->setMetadata(initialAccelsMdata);
            disconnect(accels, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        case MAG:
        {
            Magnetometer * mag = Magnetometer::GetInstance(getObjectManager());
            Q_ASSERT(mag);
            mag->setMetadata(initialMagMdata);
            disconnect(mag, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        case GYRO:
        {
            Gyros * gyros = Gyros::GetInstance(getObjectManager());
            Q_ASSERT(gyros);
            gyros->setMetadata(initialGyrosMdata);
            disconnect(gyros, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(dataUpdated(UAVObject *)));
        }
            break;

        }
    }
}

/**
 * @brief Calibration::dataUpdated Receive updates of any connected sensors and
 * process them based on the calibration state (e.g. six point or leveling)
 * @param obj The object that was updated
 */
void Calibration::dataUpdated(UAVObject * obj) {

    if (!timer.isActive()) {
        // ignore updates that come in after the timer has expired
        return;
    }

    switch(calibration_state) {
    case IDLE:
    case SIX_POINT_WAIT1:
    case SIX_POINT_WAIT2:
    case SIX_POINT_WAIT3:
    case SIX_POINT_WAIT4:
    case SIX_POINT_WAIT5:
    case SIX_POINT_WAIT6:
        // Do nothing
        return;
        break;
    case LEVELING:
        // Store data while computing the level attitude
        // and if completed go back to the idle state
        if(storeLevelingMeasurement(obj)) {
            connectSensor(GYRO, false);
            connectSensor(ACCEL, false);
            calibration_state = IDLE;

            emit showLevelingMessage(tr("Level computed"));
            emit toggleControls(true);
            emit levelingProgressChanged(0);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT1:
        // These state collect each position for six point calibration and
        // when enough data is acquired advance to the next step
        if(storeSixPointMeasurement(obj,1)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT2;
            emit showSixPointMessage(tr("Rotate left side down and press Save Position..."));
            emit toggleSavePosition(true);
            emit updatePlane(2);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT2:
        if(storeSixPointMeasurement(obj,2)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT3;
            emit showSixPointMessage(tr("Rotate upside down and press Save Position..."));
            emit toggleSavePosition(true);
            emit updatePlane(3);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT3:
        if(storeSixPointMeasurement(obj,3)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT4;
            emit showSixPointMessage(tr("Point right side down and press Save Position..."));
            emit toggleSavePosition(true);
            emit updatePlane(4);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT4:
        if(storeSixPointMeasurement(obj,4)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT5;
            emit showSixPointMessage(tr("Point nose up and press Save Position..."));
            emit toggleSavePosition(true);
            emit updatePlane(5);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT5:
        if(storeSixPointMeasurement(obj,5)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT6;
            emit showSixPointMessage(tr("Point nose down and press Save Position..."));
            emit toggleSavePosition(true);
            emit updatePlane(6);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
        }
        break;
    case SIX_POINT_COLLECT6:
        // Store data points in the final position and if enough data
        // has been computed attempt to calculate the scale and bias
        // for the accel and optionally the mag.
        if(storeSixPointMeasurement(obj,6)) {
            // All data collected.  Disconnect everything and compute value
            connectSensor(ACCEL, false);
            if (calibrateMag)
                connectSensor(MAG, false);

            calibration_state = IDLE;
            emit toggleControls(true);
            emit updatePlane(0);
            emit sixPointProgressChanged(0);
            disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

            // Do calculation
            int ret=computeScaleBias();
            if (ret==CALIBRATION_SUCCESS)
                emit showSixPointMessage(tr("Calibration succeeded"));
            else{
                //Return sensor calibration values to their original settings
                resetSensorCalibrationToOriginalValues();

                if(ret==ACCELEROMETER_FAILED){
                    emit showSixPointMessage(tr("Acceleromter calibration failed. Original values have been written back to device. Perhaps you moved too much during the calculation? Please repeat calibration."));
                }
                else if(ret==MAGNETOMETER_FAILED){
                    emit showSixPointMessage(tr("Magnetometer calibration failed. Original values have been written back to device. Perhaps you performed the calibration near iron? Please repeat calibration."));
                }
            }
        }
        break;
    }

}

/**
 * @brief Calibration::timeout When collecting data for leveling or six point calibration times out
 * clean up the state and reset
 */
void Calibration::timeout() {

    switch(calibration_state) {
    case IDLE:
        // Do nothing
        return;
        break;
    case LEVELING:
        // Disconnect appropriate sensors
        connectSensor(GYRO, false);
        connectSensor(ACCEL, false);
        calibration_state = IDLE;
        emit showLevelingMessage(tr("Leveling timed out ..."));
        emit levelingProgressChanged(0);
    case SIX_POINT_WAIT1:
    case SIX_POINT_WAIT2:
    case SIX_POINT_WAIT3:
    case SIX_POINT_WAIT4:
    case SIX_POINT_WAIT5:
    case SIX_POINT_WAIT6:
        // Do nothing, shouldn't happen
        return;
        break;
    case SIX_POINT_COLLECT1:
    case SIX_POINT_COLLECT2:
    case SIX_POINT_COLLECT3:
    case SIX_POINT_COLLECT4:
    case SIX_POINT_COLLECT5:
    case SIX_POINT_COLLECT6:
        connectSensor(ACCEL, false);
        if (calibrateMag)
            connectSensor(MAG, false);
        calibration_state = IDLE;
        emit showSixPointMessage(tr("Six point data collection timed out"));
        emit sixPointProgressChanged(0);
        break;
    }

    emit updatePlane(0);
    emit toggleControls(true);

    disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

    QMessageBox msgBox;
    msgBox.setText(tr("Calibration timed out before receiving required updates."));
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
}

/**
 * @brief Calibration::doStartLeveling Called by UI to start collecting data to calculate level
 */
void Calibration::doStartLeveling() {
    accel_accum_x.clear();
    accel_accum_y.clear();
    accel_accum_z.clear();
    gyro_accum_x.clear();
    gyro_accum_y.clear();
    gyro_accum_z.clear();

    // Disable gyro bias correction to see raw data
    AttitudeSettings *attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();
    attitudeSettingsData.BiasCorrectGyro = AttitudeSettings::BIASCORRECTGYRO_FALSE;
    attitudeSettings->setData(attitudeSettingsData);
    attitudeSettings->updated();

    calibration_state = LEVELING;

    // Connect to the sensor updates and speed them up
    connectSensor(ACCEL, true);
    connectSensor(GYRO, true);

    emit toggleControls(false);
    emit showLevelingMessage(tr("Leave board flat"));

    // Set up timeout timer
    timer.setSingleShot(true);
    timer.start(5000 + (NUM_SENSOR_UPDATES * SENSOR_UPDATE_PERIOD));
    connect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
}

/**
  * Called by the "Start" button.  Sets up the meta data and enables the
  * buttons to perform six point calibration of the magnetometer (optionally
  * accel) to compute the scale and bias of this sensor based on the current
  * home location magnetic strength.
  */
void Calibration::doStartSixPoint()
{

    // Save initial rotation settings
    AttitudeSettings * attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();

    initialBoardRotation[0]=attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL];
    initialBoardRotation[1]=attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH];
    initialBoardRotation[2]=attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW];

    //Set board rotation to (0,0,0)
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] =0;
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH]=0;
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW]  =0;
    attitudeSettings->setData(attitudeSettingsData);

    // Save initial accelerometer settings
    InertialSensorSettings * inertialSettings = InertialSensorSettings::GetInstance(getObjectManager());
    Q_ASSERT(inertialSettings);
    InertialSensorSettings::DataFields inertialSettingsData = inertialSettings->getData();

    initialAccelsScale[0]=inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_X];
    initialAccelsScale[1]=inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Y];
    initialAccelsScale[2]=inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Z];
    initialAccelsBias[0]=inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_X];
    initialAccelsBias[1]=inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Y];
    initialAccelsBias[2]=inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Z];

    // Reset the scale and bias to get a correct result
    inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_X] = 1.0;
    inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Y] = 1.0;
    inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Z] = 1.0;
    inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_X] = 0.0;
    inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Y] = 0.0;
    inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Z] = 0.0;
    inertialSettings->setData(inertialSettingsData);

    // If calibrating the mag, remove any scaling
    if (calibrateMag) {
        // Save initial magnetometer settings
        RevoCalibration *revoCalibration = RevoCalibration::GetInstance(getObjectManager());
        Q_ASSERT(revoCalibration);
        RevoCalibration::DataFields revoCalData = revoCalibration->getData();

        initialMagsScale[0]=revoCalData.MagScale[RevoCalibration::MAGSCALE_X];
        initialMagsScale[1]=revoCalData.MagScale[RevoCalibration::MAGSCALE_Y];
        initialMagsScale[2]=revoCalData.MagScale[RevoCalibration::MAGSCALE_Z];
        initialMagsBias[0]= revoCalData.MagBias[RevoCalibration::MAGBIAS_X];
        initialMagsBias[1]= revoCalData.MagBias[RevoCalibration::MAGBIAS_Y];
        initialMagsBias[2]= revoCalData.MagBias[RevoCalibration::MAGBIAS_Z];

        // Reset the scale to get a correct result
        revoCalData.MagScale[RevoCalibration::MAGSCALE_X] = 1;
        revoCalData.MagScale[RevoCalibration::MAGSCALE_Y] = 1;
        revoCalData.MagScale[RevoCalibration::MAGSCALE_Z] = 1;
        revoCalData.MagBias[RevoCalibration::MAGBIAS_X] = 0;
        revoCalData.MagBias[RevoCalibration::MAGBIAS_Y] = 0;
        revoCalData.MagBias[RevoCalibration::MAGBIAS_Z] = 0;
        revoCalibration->setData(revoCalData);
    }

    // Clear the accumulators
    accel_accum_x.clear();
    accel_accum_y.clear();
    accel_accum_z.clear();
    mag_accum_x.clear();
    mag_accum_y.clear();
    mag_accum_z.clear();

    Thread::usleep(100000);

    connectSensor(ACCEL, true);

    if(calibrateMag) {
        connectSensor(MAG, true);
    }

    // Show UI parts and update the calibration state
    emit showSixPointMessage(tr("Place horizontally and click save position..."));
    emit updatePlane(1);
    emit toggleControls(false);
    emit toggleSavePosition(true);
    calibration_state = SIX_POINT_WAIT1;
}


/**
 * @brief Calibration::doCancelSixPoint Cancels six point calibration and returns all values to their original settings.
 */
void Calibration::doCancelSixPoint(){
    //Return sensor calibration values to their original settings
    resetSensorCalibrationToOriginalValues();

    connectSensor(ACCEL, false);

    if(calibrateMag) {
        connectSensor(MAG, false);
    }

    calibration_state = IDLE;
    emit toggleControls(true);
    emit toggleSavePosition(false);
    emit updatePlane(0);
    emit sixPointProgressChanged(0);
    disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

    emit showSixPointMessage(tr("Calibration canceled."));

}


/**
  * Tells the calibration utility the UAV is in position and to collect data.
  */
void Calibration::doSaveSixPointPosition()
{
    switch(calibration_state) {
    case SIX_POINT_WAIT1:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT1;
        break;
    case SIX_POINT_WAIT2:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT2;
        break;
    case SIX_POINT_WAIT3:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT3;
        break;
    case SIX_POINT_WAIT4:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT4;
        break;
    case SIX_POINT_WAIT5:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT5;
        break;
    case SIX_POINT_WAIT6:
        emit showSixPointMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT6;
        break;
    default:
        return;
        break;
    }

    emit toggleSavePosition(false);

    // Set up timeout timer
    timer.setSingleShot(true);
    timer.start(5000 + (NUM_SENSOR_UPDATES_SIX_POINT * SENSOR_UPDATE_PERIOD));
    connect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
}


/**
 * @brief Calibration::storedLevelingMeasurement Store a measurement and if there
 * is enough data compute the level angle and gyro zero
 * @return true if enough data is collected
 */
bool Calibration::storeLevelingMeasurement(UAVObject *obj) {
    Accels * accels = Accels::GetInstance(getObjectManager());
    Gyros * gyros = Gyros::GetInstance(getObjectManager());

    // Accumulate samples until we have _at least_ NUM_SENSOR_UPDATES samples
    if(obj->getObjID() == Accels::OBJID) {
        Accels::DataFields accelsData = accels->getData();
        accel_accum_x.append(accelsData.x);
        accel_accum_y.append(accelsData.y);
        accel_accum_z.append(accelsData.z);
    } else if (obj->getObjID() == Gyros::OBJID) {
        Gyros::DataFields gyrosData = gyros->getData();
        gyro_accum_x.append(gyrosData.x);
        gyro_accum_y.append(gyrosData.y);
        gyro_accum_z.append(gyrosData.z);
    }

    // update the progress indicator
    emit levelingProgressChanged((float) qMin(accel_accum_x.size(),  gyro_accum_x.size()) / NUM_SENSOR_UPDATES * 100);

    // If we have enough samples, then stop sampling and compute the biases
    if (accel_accum_x.size() >= NUM_SENSOR_UPDATES && gyro_accum_x.size() >= NUM_SENSOR_UPDATES) {
        timer.stop();
        disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

        float x_gyro_bias = listMean(gyro_accum_x);
        float y_gyro_bias = listMean(gyro_accum_y);
        float z_gyro_bias = listMean(gyro_accum_z);

        // Get the existing attitude settings
        AttitudeSettings::DataFields attitudeSettingsData = AttitudeSettings::GetInstance(getObjectManager())->getData();
        InertialSensorSettings::DataFields inertialSensorSettingsData = InertialSensorSettings::GetInstance(getObjectManager())->getData();

        const double DEG2RAD = M_PI / 180.0f;
        const double RAD2DEG = 1.0 / DEG2RAD;
        const double GRAV = -9.81;

        // Inverse rotation of sensor data, from body frame into sensor frame
        double a_body[3] = { listMean(accel_accum_x), listMean(accel_accum_y), listMean(accel_accum_z) };
        double a_sensor[3];
        double Rsb[3][3];  // The initial board rotation
        double rpy[3] = { attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] * DEG2RAD / 100.0,
                          attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] * DEG2RAD / 100.0,
                          attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] * DEG2RAD / 100.0};
        Euler2R(rpy, Rsb);
        rotate_vector(Rsb, a_body, a_sensor, true);

        // Temporary variables
        double psi, theta, phi;

        // Keep existing yaw rotation
        psi = attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] * DEG2RAD / 100.0;

        double cP = cos(psi);
        double sP = sin(psi);

        // In case psi is too small, we have to use a different equation to solve for theta
        if (fabs(psi) > M_PI / 2)
            theta = atan((a_sensor[1] + cP * (sP * a_sensor[0] - cP * a_sensor[1])) / (sP * a_sensor[2]));
        else
            theta = atan((a_sensor[0] - sP * (sP * a_sensor[0] - cP * a_sensor[1])) / (cP * a_sensor[2]));
        phi = atan2((sP * a_sensor[0] - cP * a_sensor[1]) / GRAV, a_sensor[2] / cos(theta) / GRAV);

        attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] = phi * RAD2DEG * 100.0;
        attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] = theta * RAD2DEG * 100.0;

        // Rotate the gyro bias from the old body frame into the sensor frame
        // and then into the new body frame
        double gyro_sensor[3];
        double gyro_newbody[3];
        double gyro_oldbody[3] = {x_gyro_bias, y_gyro_bias, z_gyro_bias};
        double new_rpy[3] = { attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] * DEG2RAD / 100.0,
                              attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] * DEG2RAD / 100.0,
                              attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] * DEG2RAD / 100.0};
        rotate_vector(Rsb, gyro_oldbody, gyro_sensor, true);
        Euler2R(new_rpy, Rsb);
        rotate_vector(Rsb, gyro_sensor, gyro_newbody, false);

        // Store these new biases
        inertialSensorSettingsData.InitialGyroBias[InertialSensorSettings::INITIALGYROBIAS_X] = gyro_newbody[0];
        inertialSensorSettingsData.InitialGyroBias[InertialSensorSettings::INITIALGYROBIAS_Y] = gyro_newbody[1];
        inertialSensorSettingsData.InitialGyroBias[InertialSensorSettings::INITIALGYROBIAS_Z] = gyro_newbody[2];
        InertialSensorSettings::GetInstance(getObjectManager())->setData(inertialSensorSettingsData);

        // We offset the gyro bias by current bias to help precision
        // Disable gyro bias correction to see raw data
        AttitudeSettings *attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
        Q_ASSERT(attitudeSettings);
        attitudeSettingsData.BiasCorrectGyro = AttitudeSettings::BIASCORRECTGYRO_TRUE;
        attitudeSettings->setData(attitudeSettingsData);
        attitudeSettings->updated();

        return true;
    }
    return false;
}

/**
  * Grab a sample of accel or mag data while in this position and
  * store it for averaging.
  * @return true If enough data is averaged at this position
  */
bool Calibration::storeSixPointMeasurement(UAVObject * obj, int position)
{
    // Position is specified 1-6, but used as an index
    Q_ASSERT(position >= 1 && position <= 6);
    position --;

    if( obj->getObjID() == Accels::OBJID ) {
        Accels * accels = Accels::GetInstance(getObjectManager());
        Q_ASSERT(accels);
        Accels::DataFields accelsData = accels->getData();

        accel_accum_x.append(accelsData.x);
        accel_accum_y.append(accelsData.y);
        accel_accum_z.append(accelsData.z);
    }

    if( calibrateMag && obj->getObjID() == Magnetometer::OBJID) {
        Magnetometer * mag = Magnetometer::GetInstance(getObjectManager());
        Q_ASSERT(mag);
        Magnetometer::DataFields magData = mag->getData();

        mag_accum_x.append(magData.x);
        mag_accum_y.append(magData.y);
        mag_accum_z.append(magData.z);
    }

    emit sixPointProgressChanged((float) accel_accum_x.size() / NUM_SENSOR_UPDATES_SIX_POINT * 100);

    // If enough data is collected, average it for this position
    if(accel_accum_x.size() >= NUM_SENSOR_UPDATES_SIX_POINT &&
            (!calibrateMag || mag_accum_x.size() >= NUM_SENSOR_UPDATES_SIX_POINT)) {

        accel_data_x[position] = listMean(accel_accum_x);
        accel_data_y[position] = listMean(accel_accum_y);
        accel_data_z[position] = listMean(accel_accum_z);
        accel_accum_x.clear();
        accel_accum_y.clear();
        accel_accum_z.clear();

        if (calibrateMag) {
            mag_data_x[position] = listMean(mag_accum_x);
            mag_data_y[position] = listMean(mag_accum_y);
            mag_data_z[position] = listMean(mag_accum_z);
            mag_accum_x.clear();
            mag_accum_y.clear();
            mag_accum_z.clear();
        }

        // Indicate all data collected for this position
        return true;
    }
    return false;
}

/**
 * Util function to get a pointer to the object manager
 * @return pointer to the UAVObjectManager
 */
UAVObjectManager* Calibration::getObjectManager() {
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objMngr);
    return objMngr;
}

/**
 * Utility function which calculates the Mean value of a list of values
 * @param list list of double values
 * @returns Mean value of the list of parameter values
 */
double Calibration::listMean(QList<double> list)
{
    double accum = 0;
    for(int i = 0; i < list.size(); i++)
        accum += list[i];
    return accum / list.size();
}

/**
  * Computes the scale and bias for the accelerometer and mag once all the data
  * has been collected in 6 positions.
  */
int Calibration::computeScaleBias()
{
    // Regardless of calibration result, set board rotations back to user settings
    AttitudeSettings * attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] = initialBoardRotation[0];
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] = initialBoardRotation[1];
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] = initialBoardRotation[2];
    attitudeSettings->setData(attitudeSettingsData);

    bool good_calibration = true;

    // Calibrate accelerometer
    double S[3], b[3];
    SixPointInConstFieldCal(accelLength, accel_data_x, accel_data_y, accel_data_z, S, b);

    InertialSensorSettings * inertialSensorSettings = InertialSensorSettings::GetInstance(getObjectManager());
    Q_ASSERT(inertialSensorSettings);
    InertialSensorSettings::DataFields inertialSensorSettingsData = inertialSensorSettings->getData();

    //Assign calibration data
    inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_X] += (-sign(S[0]) * b[0]);
    inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Y] += (-sign(S[1]) * b[1]);
    inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Z] += (-sign(S[2]) * b[2]);

    inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_X] *= fabs(S[0]);
    inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Y] *= fabs(S[1]);
    inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Z] *= fabs(S[2]);

    // Check the accel calibration is good
    good_calibration &= inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_X] ==
            inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_X];
    good_calibration &= inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Y] ==
            inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Y];
    good_calibration &= inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Z] ==
            inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Z];
    good_calibration &= (inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_X] ==
                         inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_X]);
    good_calibration &= (inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Y] ==
                         inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Y]);
    good_calibration &= (inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Z] ==
                         inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Z]);

    //This can happen if, for instance, HomeLocation.g_e == 0
    if((S[0]+S[1]+S[2])<0.0001){
        good_calibration=false;
    }

    if (calibrateMag) {
        good_calibration = true;

        // Work out the average vector length as nominal since mag scale normally close to 1
        // and we don't use the magnitude anyway.  Avoids requiring home location.
        double m_x = 0, m_y = 0, m_z = 0, len = 0;
        for (int i = 0; i < 6; i++) {
            m_x += mag_data_x[i];
            m_y += mag_data_x[i];
            m_z += mag_data_x[i];
        }
        m_x /= 6;
        m_y /= 6;
        m_z /= 6;
        for (int i = 0; i < 6; i++) {
            len += sqrt(pow(mag_data_x[i] - m_x,2) + pow(mag_data_y[i] - m_y,2) + pow(mag_data_z[i] - m_z,2));
        }
        len /= 6;

        // Calibrate magnetomter
        double S[3], b[3];
        SixPointInConstFieldCal(len, mag_data_x, mag_data_y, mag_data_z, S, b);

        RevoCalibration * revoCalibration = RevoCalibration::GetInstance(getObjectManager());
        Q_ASSERT(revoCalibration);
        RevoCalibration::DataFields revoCalibrationData = revoCalibration->getData();

        //Assign calibration data
        revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_X] += (-sign(S[0]) * b[0]);
        revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_Y] += (-sign(S[1]) * b[1]);
        revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_Z] += (-sign(S[2]) * b[2]);
        revoCalibrationData.MagScale[RevoCalibration::MAGSCALE_X] *= fabs(S[0]);
        revoCalibrationData.MagScale[RevoCalibration::MAGSCALE_Y] *= fabs(S[1]);
        revoCalibrationData.MagScale[RevoCalibration::MAGSCALE_Z] *= fabs(S[2]);

        // Check the mag calibration is good
        good_calibration &= revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_X] ==
                revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_X];
        good_calibration &= revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_Y] ==
                revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_Y];
        good_calibration &= revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_Z]  ==
                revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_Z] ;
        good_calibration &= revoCalibrationData.MagScale[RevoCalibration::MAGSCALE_X] ==
                revoCalibrationData.MagScale[RevoCalibration::MAGSCALE_X];
        good_calibration &= revoCalibrationData.MagScale[RevoCalibration::MAGSCALE_Y] ==
                revoCalibrationData.MagScale[RevoCalibration::MAGSCALE_Y];
        good_calibration &= revoCalibrationData.MagScale[RevoCalibration::MAGSCALE_Z] ==
                revoCalibrationData.MagScale[RevoCalibration::MAGSCALE_Z];

        //This can happen if, for instance, HomeLocation.g_e == 0
        if((S[0]+S[1]+S[2])<0.0001){
            good_calibration=false;
        }

        if (good_calibration) {
            qDebug()<<  "Mag bias: " << revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_X] << " " << revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_Y]  << " " << revoCalibrationData.MagBias[RevoCalibration::MAGBIAS_Z];
            revoCalibration->setData(revoCalibrationData);
        } else {
            return MAGNETOMETER_FAILED;
        }
    }

    // Apply at the end so only applies if it works and mag does too
    if (good_calibration) {
        qDebug()<<  "Accel bias: " << inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_X] << " " << inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Y] << " " << inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Z];
        inertialSensorSettings->setData(inertialSensorSettingsData);
    } else {
        return ACCELEROMETER_FAILED;
    }

    return CALIBRATION_SUCCESS;
}

/**
 * @brief Calibration::resetToOriginalValues Resets the accelerometer and magnetometer setting to their pre-calibration values
 */
void Calibration::resetSensorCalibrationToOriginalValues()
{

    // Write original board rotation settings back to device
    AttitudeSettings * attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();

    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] = initialBoardRotation[0];
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] = initialBoardRotation[1];
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] = initialBoardRotation[2];
    attitudeSettings->setData(attitudeSettingsData);


    //Write the original accelerometer values back to the device
    InertialSensorSettings * inertialSettings = InertialSensorSettings::GetInstance(getObjectManager());
    Q_ASSERT(inertialSettings);
    InertialSensorSettings::DataFields inertialSettingsData = inertialSettings->getData();

    inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_X]=initialAccelsScale[0];
    inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Y]=initialAccelsScale[1];
    inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Z]=initialAccelsScale[2];
    inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_X]=initialAccelsBias[0];
    inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Y]=initialAccelsBias[1];
    inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Z]=initialAccelsBias[2];

    inertialSettings->setData(inertialSettingsData);

    if (calibrateMag) {
        //Write the original magnetometer values back to the device
        RevoCalibration *revoCalibration = RevoCalibration::GetInstance(getObjectManager());
        Q_ASSERT(revoCalibration);
        RevoCalibration::DataFields revoCalData = revoCalibration->getData();

        revoCalData.MagScale[RevoCalibration::MAGSCALE_X]=initialMagsScale[0];
        revoCalData.MagScale[RevoCalibration::MAGSCALE_Y]=initialMagsScale[1];
        revoCalData.MagScale[RevoCalibration::MAGSCALE_Z]=initialMagsScale[2];
        revoCalData.MagBias[RevoCalibration::MAGBIAS_X]=initialMagsBias[0];
        revoCalData.MagBias[RevoCalibration::MAGBIAS_Y]=initialMagsBias[1];
        revoCalData.MagBias[RevoCalibration::MAGBIAS_Z]=initialMagsBias[2];

        revoCalibration->setData(revoCalData);
    }


}

/**
 * @brief Compute a rotation matrix from a set of euler angles
 * @param[in] rpy The euler angles in roll, pitch, yaw
 * @param[out] Rbe The rotation matrix to take a matrix from
 *       0,0,0 to that rotation
 */
void Calibration::Euler2R(double rpy[3], double Rbe[3][3])
{
    double sF = sin(rpy[0]), cF = cos(rpy[0]);
    double sT = sin(rpy[1]), cT = cos(rpy[1]);
    double sP = sin(rpy[2]), cP = cos(rpy[2]);

    Rbe[0][0] = cT*cP;
    Rbe[0][1] = cT*sP;
    Rbe[0][2] = -sT;
    Rbe[1][0] = sF*sT*cP - cF*sP;
    Rbe[1][1] = sF*sT*sP + cF*cP;
    Rbe[1][2] = cT*sF;
    Rbe[2][0] = cF*sT*cP + sF*sP;
    Rbe[2][1] = cF*sT*sP - sF*cP;
    Rbe[2][2] = cT*cF;
}

/**
 * @brief Rotate a vector by the rotation matrix, optionally trasposing
 * @param[in] R the rotation matrix
 * @param[in] vec The vector to rotate by this matrix
 * @param[out] vec_out The rotated vector
 * @param[in] transpose Optionally transpose the rotation matrix first (reverse the rotation)
 */
void Calibration::rotate_vector(double R[3][3], const double vec[3], double vec_out[3], bool transpose = true)
{
    if (!transpose){
        vec_out[0] = R[0][0] * vec[0] + R[0][1] * vec[1] + R[0][2] * vec[2];
        vec_out[1] = R[1][0] * vec[0] + R[1][1] * vec[1] + R[1][2] * vec[2];
        vec_out[2] = R[2][0] * vec[0] + R[2][1] * vec[1] + R[2][2] * vec[2];
    }
    else {
        vec_out[0] = R[0][0] * vec[0] + R[1][0] * vec[1] + R[2][0] * vec[2];
        vec_out[1] = R[0][1] * vec[0] + R[1][1] * vec[1] + R[2][1] * vec[2];
        vec_out[2] = R[0][2] * vec[0] + R[1][2] * vec[1] + R[2][2] * vec[2];
    }
}

/**
 * @name LinearEquationsSolving
 * @brief Uses Gaussian Elimination to solve linear equations of the type Ax=b
 *
 * @note Matrix code snarfed from: http://www.hlevkin.com/NumAlg/LinearEquations.c
 *
 * @return 0 if system not solving
 * @param[in] nDim - system dimension
 * @param[in] pfMatr - matrix with coefficients
 * @param[in] pfVect - vector with free members
 * @param[out] pfSolution - vector with system solution
 * @param[out] pfMatr becames trianglular after function call
 * @param[out] pfVect changes after function call
 * @author Henry Guennadi Levkin
 */
int Calibration::LinearEquationsSolving(int nDim, double* pfMatr, double* pfVect, double* pfSolution)
{
    double fMaxElem;
    double fAcc;

    int i , j, k, m;

    for(k=0; k<(nDim-1); k++) // base row of matrix
    {
        // search of line with max element
        fMaxElem = fabs( pfMatr[k*nDim + k] );
        m = k;
        for(i=k+1; i<nDim; i++)
        {
            if(fMaxElem < fabs(pfMatr[i*nDim + k]) )
            {
                fMaxElem = pfMatr[i*nDim + k];
                m = i;
            }
        }

        // permutation of base line (index k) and max element line(index m)
        if(m != k)
        {
            for(i=k; i<nDim; i++)
            {
                fAcc               = pfMatr[k*nDim + i];
                pfMatr[k*nDim + i] = pfMatr[m*nDim + i];
                pfMatr[m*nDim + i] = fAcc;
            }
            fAcc = pfVect[k];
            pfVect[k] = pfVect[m];
            pfVect[m] = fAcc;
        }

        if( pfMatr[k*nDim + k] == 0.) return 0; // needs improvement !!!

        // triangulation of matrix with coefficients
        for(j=(k+1); j<nDim; j++) // current row of matrix
        {
            fAcc = - pfMatr[j*nDim + k] / pfMatr[k*nDim + k];
            for(i=k; i<nDim; i++)
            {
                pfMatr[j*nDim + i] = pfMatr[j*nDim + i] + fAcc*pfMatr[k*nDim + i];
            }
            pfVect[j] = pfVect[j] + fAcc*pfVect[k]; // free member recalculation
        }
    }

    for(k=(nDim-1); k>=0; k--)
    {
        pfSolution[k] = pfVect[k];
        for(i=(k+1); i<nDim; i++)
        {
            pfSolution[k] -= (pfMatr[k*nDim + i]*pfSolution[i]);
        }
        pfSolution[k] = pfSolution[k] / pfMatr[k*nDim + k];
    }

    return 1;
}

/**
 * @name SixPointInConstFieldCal
 * @brief Compute the scale and bias assuming the data comes from six orientations
 *        in a constant field
 *
 *   x, y, z are vectors of six measurements
 *
 * Computes sensitivity and offset such that:
 *
 * c = S * A + b
 *
 * where c is the measurement, S is the sensitivity, b is the bias offset, and
 * A is the field being measured  expressed as a ratio of the measured value
 * to the field strength. aka a direction cosine.
 *
 * A is what we really want and it is computed using the equation:
 *
 * A = (c - b)/S
 *
 */
int Calibration::SixPointInConstFieldCal( double ConstMag, double x[6], double y[6], double z[6], double S[3], double b[3] )
{
    int i;
    double A[5][5];
    double f[5], c[5];
    double xp, yp, zp, Sx;

    // Fill in matrix A -
    // write six difference-in-magnitude equations of the form
    // Sx^2(x2^2-x1^2) + 2*Sx*bx*(x2-x1) + Sy^2(y2^2-y1^2) + 2*Sy*by*(y2-y1) + Sz^2(z2^2-z1^2) + 2*Sz*bz*(z2-z1) = 0
    // or in other words
    // 2*Sx*bx*(x2-x1)/Sx^2  + Sy^2(y2^2-y1^2)/Sx^2  + 2*Sy*by*(y2-y1)/Sx^2  + Sz^2(z2^2-z1^2)/Sx^2  + 2*Sz*bz*(z2-z1)/Sx^2  = (x1^2-x2^2)
    for (i=0;i<5;i++){
        A[i][0] = 2.0 * (x[i+1] - x[i]);
        A[i][1] = y[i+1]*y[i+1] - y[i]*y[i];
        A[i][2] = 2.0 * (y[i+1] - y[i]);
        A[i][3] = z[i+1]*z[i+1] - z[i]*z[i];
        A[i][4] = 2.0 * (z[i+1] - z[i]);
        f[i]    = x[i]*x[i] - x[i+1]*x[i+1];
    }

    // solve for c0=bx/Sx, c1=Sy^2/Sx^2; c2=Sy*by/Sx^2, c3=Sz^2/Sx^2, c4=Sz*bz/Sx^2
    if (  !LinearEquationsSolving( 5, (double *)A, f, c) ) return 0;

    // use one magnitude equation and c's to find Sx - doesn't matter which - all give the same answer
    xp = x[0]; yp = y[0]; zp = z[0];
    Sx = sqrt(ConstMag*ConstMag / (xp*xp + 2*c[0]*xp + c[0]*c[0] + c[1]*yp*yp + 2*c[2]*yp + c[2]*c[2]/c[1] + c[3]*zp*zp + 2*c[4]*zp + c[4]*c[4]/c[3]));

    S[0] = Sx;
    b[0] = Sx*c[0];
    S[1] = sqrt(c[1]*Sx*Sx);
    b[1] = c[2]*Sx*Sx/S[1];
    S[2] = sqrt(c[3]*Sx*Sx);
    b[2] = c[4]*Sx*Sx/S[2];

    return 1;
}
