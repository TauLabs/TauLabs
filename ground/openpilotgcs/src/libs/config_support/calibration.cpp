/**
 ******************************************************************************
 *
 * @file       calibration.cpp
 * @author     The PhoenixPilot Team, http://www.openpilot.org Copyright (C) 2012.
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
#include "configccattitudewidget.h"
#include "ui_ccattitude.h"
#include "utils/coordinateconversions.h"
#include <QMutexLocker>
#include <QMessageBox>
#include <QDebug>
#include <QDesktopServices>
#include <QUrl>
#include "accels.h"
#include "gyros.h"
#include "magnetometer.h"
#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>
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

#define sign(x) ((x < 0) ? -1 : 1)

Calibration::Calibration() : calibrate_mag(false), magLength(100), accelLength(9.81)
{
}

Calibration::~Calibration()
{
}

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
        return;
        break;
    case LEVELING:
        break;
    case SIX_POINT_COLLECT1:
        if(storeSixPointMeasured(obj,1)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT2;
            emit showMessage(tr("Rotate UAV and press save position..."));
            emit toggleControls(true);
            emit updatePlane(2);
        }
        break;
    case SIX_POINT_COLLECT2:
        if(storeSixPointMeasured(obj,2)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT3;
            emit showMessage(tr("Rotate UAV and press save position..."));
            emit toggleControls(true);
            emit updatePlane(3);
        }
        break;
    case SIX_POINT_COLLECT3:
        if(storeSixPointMeasured(obj,3)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT4;
            emit showMessage(tr("Rotate UAV and press save position..."));
            emit toggleControls(true);
            emit updatePlane(4);
        }
        break;
    case SIX_POINT_COLLECT4:
        if(storeSixPointMeasured(obj,4)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT3;
            emit showMessage(tr("Rotate UAV and press save position..."));
            emit toggleControls(true);
            emit updatePlane(5);
        }
        break;
    case SIX_POINT_COLLECT5:
        if(storeSixPointMeasured(obj,5)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT3;
            emit showMessage(tr("Rotate UAV and press save position..."));
            emit toggleControls(true);
            emit updatePlane(6);
        }
        break;
    case SIX_POINT_COLLECT6:
        if(storeSixPointMeasured(obj,6)) {
            // If all the data is collected advance to the next position
            calibration_state = SIX_POINT_WAIT3;
            emit toggleControls(true);
            emit updatePlane(0);
            computeScaleBias();

            // TODO: Disconnect mag if appropriate, restore meta data
            Accels * accels = Accels::GetInstance(getObjectManager());
            Q_ASSERT(accels);
            disconnect(accels,SIGNAL(objectUpdated(UAVObject*)),this,SLOT(dataUpdated(UAVObject*)));


        }
        break;
    }

    Accels * accels = Accels::GetInstance(getObjectManager());
    Gyros * gyros = Gyros::GetInstance(getObjectManager());

    // Accumulate samples until we have _at least_ NUM_SENSOR_UPDATES samples
    // for both gyros and accels.
    // Note that, at present, we stash the samples and then compute the bias
    // at the end, even though the mean could be accumulated as we go.
    // In future, a better algorithm could be used.
    if(obj->getObjID() == Accels::OBJID) {
        accelUpdates++;
        Accels::DataFields accelsData = accels->getData();
        accel_accum_x.append(accelsData.x);
        accel_accum_y.append(accelsData.y);
        accel_accum_z.append(accelsData.z);
    } else if (obj->getObjID() == Gyros::OBJID) {
        gyroUpdates++;
        Gyros::DataFields gyrosData = gyros->getData();
        gyro_accum_x.append(gyrosData.x);
        gyro_accum_y.append(gyrosData.y);
        gyro_accum_z.append(gyrosData.z);
    }

    // update the progress indicator
    ui->zeroBiasProgress->setValue((float) qMin(accelUpdates, gyroUpdates) / NUM_SENSOR_UPDATES * 100);

    // If we have enough samples, then stop sampling and compute the biases
    if (accelUpdates >= NUM_SENSOR_UPDATES && gyroUpdates >= NUM_SENSOR_UPDATES) {
        timer.stop();
        disconnect(obj,SIGNAL(objectUpdated(UAVObject*)),this,SLOT(sensorsUpdated(UAVObject*)));
        disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

        float x_gyro_bias = listMean(gyro_accum_x);
        float y_gyro_bias = listMean(gyro_accum_y);
        float z_gyro_bias = listMean(gyro_accum_z);
        accels->setMetadata(initialAccelsMdata);
        gyros->setMetadata(initialGyrosMdata);

        // Get the existing attitude settings
        AttitudeSettings::DataFields attitudeSettingsData = AttitudeSettings::GetInstance(getObjectManager())->getData();
        InertialSensorSettings::DataFields inertialSensorSettingsData = InertialSensorSettings::GetInstance(getObjectManager())->getData();

        const double DEG2RAD = M_PI / 180.0f;
        const double RAD2DEG = 1.0 / DEG2RAD;
        const double GRAV = -9.81;

        // Inverse rotation of sensor data, from body frame into sensor frame
        double a_body[3] = { listMean(accel_accum_x), listMean(accel_accum_y), listMean(accel_accum_z) };
        double a_sensor[3];
        double Rsb[3][3];
        double rpy[3] = { attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] * DEG2RAD / 100.0,
                          attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] * DEG2RAD / 100.0,
                          attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] * DEG2RAD / 100.0};
        Euler2R(rpy, Rsb);
        rot_mult(Rsb, a_body, a_sensor, true);

        qDebug() << "A before: " << a_body[0] << " " << a_body[1] << " " << a_body[2];
        qDebug() << "A after: " << a_sensor[0] << " " << a_sensor[1] << " " << a_sensor[2];

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

        inertialSensorSettingsData.InitialGyroBias[InertialSensorSettings::INITIALGYROBIAS_X] = x_gyro_bias;
        inertialSensorSettingsData.InitialGyroBias[InertialSensorSettings::INITIALGYROBIAS_Y] = y_gyro_bias;
        inertialSensorSettingsData.InitialGyroBias[InertialSensorSettings::INITIALGYROBIAS_Z] = z_gyro_bias;
        InertialSensorSettings::GetInstance(getObjectManager())->setData(inertialSensorSettingsData);
        InertialSensorSettings::GetInstance(getObjectManager())->updated();

        // We offset the gyro bias by current bias to help precision
        // Disable gyro bias correction to see raw data
        AttitudeSettings *attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
        Q_ASSERT(attitudeSettings);
        attitudeSettingsData.BiasCorrectGyro = AttitudeSettings::BIASCORRECTGYRO_TRUE;
        attitudeSettings->setData(attitudeSettingsData);
        attitudeSettings->updated();

        this->setDirty(true);

        // reenable controls
        enableControls(true);
    }
}

void Calibration::timeout() {
    UAVDataObject * obj = Accels::GetInstance(getObjectManager());
    disconnect(obj,SIGNAL(objectUpdated(UAVObject*)),this,SLOT(sensorsUpdated(UAVObject*)));
    disconnect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));

    Accels * accels = Accels::GetInstance(getObjectManager());
    Gyros * gyros = Gyros::GetInstance(getObjectManager());
    accels->setMetadata(initialAccelsMdata);
    gyros->setMetadata(initialGyrosMdata);

    QMessageBox msgBox;
    msgBox.setText(tr("Calibration timed out before receiving required updates."));
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();

    // reset progress indicator
    ui->zeroBiasProgress->setValue(0);
    // reenable controls
    enableControls(true);
}

void Calibration::doStartLeveling() {
    // disable controls during sampling
    enableControls(false);

    accelUpdates = 0;
    gyroUpdates = 0;
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

    // Set up to receive updates
    UAVDataObject * accels = Accels::GetInstance(getObjectManager());
    UAVDataObject * gyros = Gyros::GetInstance(getObjectManager());
    connect(accels,SIGNAL(objectUpdated(UAVObject*)),this,SLOT(sensorsUpdated(UAVObject*)));
    connect(gyros,SIGNAL(objectUpdated(UAVObject*)),this,SLOT(sensorsUpdated(UAVObject*)));

    // Speed up updates
    initialAccelsMdata = accels->getMetadata();
    UAVObject::Metadata accelsMdata = initialAccelsMdata;
    UAVObject::SetFlightTelemetryUpdateMode(accelsMdata, UAVObject::UPDATEMODE_PERIODIC);
    accelsMdata.flightTelemetryUpdatePeriod = 30; // ms
    accels->setMetadata(accelsMdata);

    initialGyrosMdata = gyros->getMetadata();
    UAVObject::Metadata gyrosMdata = initialGyrosMdata;
    UAVObject::SetFlightTelemetryUpdateMode(gyrosMdata, UAVObject::UPDATEMODE_PERIODIC);
    gyrosMdata.flightTelemetryUpdatePeriod = 30; // ms
    gyros->setMetadata(gyrosMdata);

    // Set up timeout timer
    timer.setSingleShot(true);
    timer.start(5000 + (NUM_SENSOR_UPDATES * qMax(accelsMdata.flightTelemetryUpdatePeriod,
                                                  gyrosMdata.flightTelemetryUpdatePeriod)));
    connect(&timer,SIGNAL(timeout()),this,SLOT(timeout()));
}

void Calibration::openHelp()
{

    QDesktopServices::openUrl( QUrl("http://wiki.openpilot.org/x/44Cf", QUrl::StrictMode) );
}

void Calibration::enableControls(bool enable)
{
    if(ui->zeroBias)
        ui->zeroBias->setEnabled(enable);
    ConfigTaskWidget::enableControls(enable);
}

void Calibration::updateObjectsFromWidgets()
{
    ConfigTaskWidget::updateObjectsFromWidgets();

    ui->zeroBiasProgress->setValue(0);
}


/********** Functions for six point calibration **************/

/**
  * Called by the "Start" button.  Sets up the meta data and enables the
  * buttons to perform six point calibration of the magnetometer (optionally
  * accel) to compute the scale and bias of this sensor based on the current
  * home location magnetic strength.
  */
void Calibration::doStartSixPoint()
{
    AttitudeSettings * attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
    Q_ASSERT(attitudeSettings);
    AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();

    //Save board rotation settings
    boardRotation[0]=attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL];
    boardRotation[1]=attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH];
    boardRotation[2]=attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW];

    //Set board rotation to (0,0,0)
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] =0;
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH]=0;
    attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW]  =0;

    accel_accum_x.clear();
    accel_accum_y.clear();
    accel_accum_z.clear();

    attitudeSettings->setData(attitudeSettingsData);

    // Must reset the scale to get a correct result
    InertialSensorSettings * inertialSettings = InertialSensorSettings::GetInstance(getObjectManager());
    Q_ASSERT(inertialSettings);
    InertialSensorSettings::DataFields inertialSettingsData = inertialSettings->getData();
    inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_X] = 1.0;
    inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Y] = 1.0;
    inertialSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Z] = 1.0;
    inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_X] = 0.0;
    inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Y] = 0.0;
    inertialSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Z] = 0.0;
    inertialSettings->setData(inertialSettingsData);

    Thread::usleep(100000);

    gyro_accum_x.clear();
    gyro_accum_y.clear();
    gyro_accum_z.clear();

    UAVObject::Metadata mdata;

    /* Need to get as many accel updates as possible */
    Accels * accels = Accels::GetInstance(getObjectManager());
    Q_ASSERT(accels);

    initialAccelsMdata = accels->getMetadata();
    mdata = initialAccelsMdata;
    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
    mdata.flightTelemetryUpdatePeriod = 50;
    accels->setMetadata(mdata);

    // Show UI parts and update the calibration state
    emit showMessage(tr("Place horizontally and click save position..."));
    emit updatePlane(1);
    emit toggleControls(true);
    calibration_state = SIX_POINT_WAIT1;
}


/**
  * Tells the calibration utility the UAV is in position and to collect data.
  */
void Calibration::doSaveSixPointPosition()
{
    switch(calibration_state) {
    case SIX_POINT_WAIT1:
        emit showMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT1;
        break;
    case SIX_POINT_WAIT2:
        emit showMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT2;
        break;
    case SIX_POINT_WAIT3:
        emit showMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT3;
        break;
    case SIX_POINT_WAIT4:
        emit showMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT4;
        break;
    case SIX_POINT_WAIT5:
        emit showMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT5;
        break;
    case SIX_POINT_WAIT6:
        emit showMessage(tr("Hold..."));
        emit toggleControls(false);
        calibration_state = SIX_POINT_COLLECT6;
        break;
    default:
        break;
    }
}

/**
  * Grab a sample of accel or mag data while in this position and
  * store it for averaging.
  * @return true If enough data is averaged at this position
  */
void Calibration::storeSixPointMeasured(UAVObject * obj, int position)
{
    // Position is specified 1-6, but used as an index
    Q_ASSERT(position >= 1 && position < 6);
    position --;

    if( obj->getObjID() == Accels::OBJID ) {
        Accels * accels = Accels::GetInstance(getObjectManager());
        Q_ASSERT(accels);
        Accels::DataFields accelsData = accels->getData();

        accel_accum_x.append(accelsData.x);
        accel_accum_y.append(accelsData.y);
        accel_accum_z.append(accelsData.z);
    } else {
        Q_ASSERT(0);
    }

    if( obj->getObjID() == Magnetometer::OBJID && calibrateMag) {
        Magnetometer * mag = Magnetometer::GetInstance(getObjectManager());
        Q_ASSERT(mag);
        Magnetometer::DataFields magData = mag->getData();

        mag_data_accum_x.append(magData.x);
        mag_data_accum_y.append(magData.y);
        mag_data_accum_z.append(magData.z);
    } else {
        Q_ASSERT(0);
    }

    if(accel_accum_x.size() >= NUM_SENSOR_UPDATES_SIX_POINT &&
            (!calibrateMag || mag_accum_x.size() >= NUM_SENSOR_UPDATES_SIX_POINT)) {

        accel_data_x[position] = listMean(accel_accum_x);
        accel_data_y[position] = listMean(accel_accum_y);
        accel_data_z[position] = listMean(accel_accum_z);

        if (calibrateMag) {
            mag_data_x[position] = listMean(mag_accum_x);
            mag_data_y[position] = listMean(mag_accum_y);
            mag_data_z[position] = listMean(mag_accum_z);
        }

        // Indicate all data collected for this position
        return true;
    }
    return false;
}

/**
  * Computes the scale and bias for the accelerometer once all the data
  * has been collected in 6 positions.
  */
void Calibration::computeScaleBias()
{
   double S[3], b[3];
   InertialSensorSettings * inertialSensorSettings = InertialSensorSettings::GetInstance(getObjectManager());
   HomeLocation * homeLocation = HomeLocation::GetInstance(getObjectManager());
   Q_ASSERT(inertialSensorSettings);
   Q_ASSERT(homeLocation);
   InertialSensorSettings::DataFields inertialSensorSettingsData = inertialSensorSettings->getData();
   HomeLocation::DataFields homeLocationData = homeLocation->getData();

   // Calibrate accelerometer
   SixPointInConstFieldCal2( homeLocationData.g_e, accel_data_x, accel_data_y, accel_data_z, S, b);

   //Assign calibration data
   inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_X] += (-sign(S[0]) * b[0]);
   inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Y] += (-sign(S[1]) * b[1]);
   inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Z] += (-sign(S[2]) * b[2]);

   inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_X] *= fabs(S[0]);
   inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Y] *= fabs(S[1]);
   inertialSensorSettingsData.AccelScale[InertialSensorSettings::ACCELSCALE_Z] *= fabs(S[2]);

   // Check the accel calibration is good
   bool good_calibration = true;
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

   if (good_calibration) {
       inertialSensorSettings->setData(inertialSensorSettingsData);
       this->setDirty(true);

       ui->sixPointCalibInstructions->append("Successfully computed accelerometer bias");
   } else {
       inertialSensorSettingsData = inertialSensorSettings->getData();
       ui->sixPointCalibInstructions->append("Bad calibration. Please repeat.");
   }

   //Set board rotations back to user settings
   AttitudeSettings * attitudeSettings = AttitudeSettings::GetInstance(getObjectManager());
   Q_ASSERT(attitudeSettings);
   AttitudeSettings::DataFields attitudeSettingsData = attitudeSettings->getData();
   attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_ROLL] = boardRotation[0];
   attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_PITCH] = boardRotation[1];
   attitudeSettingsData.BoardRotation[AttitudeSettings::BOARDROTATION_YAW] = boardRotation[2];
   attitudeSettings->setData(attitudeSettingsData);

   qDebug()<<  "S: " << S[0] << " " << S[1] << " " << S[2];
   qDebug()<<  "b: " << b[0] << " " << b[1] << " " << b[2];
   qDebug()<<  "Accel bias: " << inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_X] << " " << inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Y] << " " << inertialSensorSettingsData.AccelBias[InertialSensorSettings::ACCELBIAS_Z];


   position = -1; //set to run again
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
int Calibration::LinearEquationsSolving2(int nDim, double* pfMatr, double* pfVect, double* pfSolution)
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
int Calibration::SixPointInConstFieldCal2( double ConstMag, double x[6], double y[6], double z[6], double S[3], double b[3] )
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
    if (  !LinearEquationsSolving2( 5, (double *)A, f, c) ) return 0;

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
