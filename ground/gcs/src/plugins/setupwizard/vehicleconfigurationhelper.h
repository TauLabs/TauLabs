/**
 ******************************************************************************
 *
 * @file       vehicleconfigurationhelper.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup SetupWizard Setup Wizard
 * @{
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

#ifndef VEHICLECONFIGURATIONHELPER_H
#define VEHICLECONFIGURATIONHELPER_H

#include <QList>
#include <QPair>
#include "vehicleconfigurationsource.h"
#include "uavobjectmanager.h"
#include "systemsettings.h"
#include "cfg_vehicletypes/vehicleconfig.h"
#include "actuatorsettings.h"

struct mixerChannelSettings {
    int type;
    int throttle1;
    int throttle2;
    int roll;
    int pitch;
    int yaw;

    mixerChannelSettings() : type(), throttle1(), throttle2(), roll(), pitch(), yaw() {}

    mixerChannelSettings(int t, int th1, int th2, int r, int p, int y)
        : type(t), throttle1(th1), throttle2(th2), roll(r), pitch(p), yaw(y) {}
};

/**
 * @brief The VehicleConfigurationHelper class provides an interface between
 * the settings selected in the wizard and storing them on the FC.
 *
 * It will store all the options the user selects in the wizard and then in one
 * step can apply and save all of these.  When appropriate it delegates specific
 * board type details to the board plugin and should not contain any board-specific
 * code.
 */
class VehicleConfigurationHelper : public QObject {
    Q_OBJECT

public:
    VehicleConfigurationHelper(VehicleConfigurationSource *configSource);
    bool setupVehicle(bool save = true);
    bool setupHardwareSettings(bool save = true);
    static const qint16 LEGACY_ESC_FREQUENCE;
    static const qint16 RAPID_ESC_FREQUENCE;

signals:
    void saveProgress(int total, int current, QString description);

private:
    static const int MIXER_TYPE_DISABLED = 0;
    static const int MIXER_TYPE_MOTOR    = 1;
    static const int MIXER_TYPE_SERVO    = 2;
    static const float DEFAULT_ENABLED_ACCEL_TAU = 0.1;

    VehicleConfigurationSource *m_configSource;
    UAVObjectManager *m_uavoManager;

    QList<QPair<UAVDataObject *, QString> * > m_modifiedObjects;
    void addModifiedObject(UAVDataObject *object, QString description);
    void clearModifiedObjects();

    void applyHardwareConfiguration();
    void applyVehicleConfiguration();
    void applyActuatorConfiguration();
    void applyFlighModeConfiguration();
    void applySensorBiasConfiguration();
    void applyStabilizationConfiguration();
    void applyManualControlDefaults();

    void applyMixerConfiguration(mixerChannelSettings channels[]);

    GUIConfigDataUnion getGUIConfigData();
    void applyMultiGUISettings(SystemSettings::AirframeTypeOptions airframe, GUIConfigDataUnion guiConfig);

    bool saveChangesToController(bool save);
    QEventLoop m_eventLoop;
    bool m_transactionOK;
    bool m_transactionTimeout;
    int m_currentTransactionObjectID;
    int m_progress;

    void resetVehicleConfig();
    void resetGUIData();

    void setupTriCopter();
    void setupQuadCopter();
    void setupHexaCopter();
    void setupOctoCopter();

private slots:
    void uAVOTransactionCompleted(UAVObject *object, bool success);
    void uAVOTransactionCompleted(int oid, bool success);
    void saveChangesTimeout();
};

#endif // VEHICLECONFIGURATIONHELPER_H
