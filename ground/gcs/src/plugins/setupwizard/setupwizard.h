/**
 ******************************************************************************
 *
 * @file       setupwizard.h
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

#ifndef SETUPWIZARD_H
#define SETUPWIZARD_H

#include <QWizard>
#include <coreplugin/icore.h>
#include <coreplugin/iboardtype.h>
#include <coreplugin/connectionmanager.h>
#include "vehicleconfigurationsource.h"
#include "vehicleconfigurationhelper.h"

/**
 * @brief The SetupWizard class is the main interface to the setup
 * wizard.  It provides selects the sequence of pages that are presented.
 *
 * Some of the page sequences are determined by the properties of the
 * board, such as whether the input page is supported or the board can
 * be updated.
 */
class SetupWizard : public QWizard, public VehicleConfigurationSource {
    Q_OBJECT

public:
    SetupWizard(QWidget *parent = 0);
    int nextId() const;

    void setControllerType(Core::IBoardType* type)
    {
        m_controllerType = type;
    }
    Core::IBoardType* getControllerType() const
    {
        return m_controllerType;
    }

    void setVehicleType(SetupWizard::VEHICLE_TYPE type)
    {
        m_vehicleType = type;
    }
    SetupWizard::VEHICLE_TYPE getVehicleType() const
    {
        return m_vehicleType;
    }

    void setVehicleSubType(SetupWizard::VEHICLE_SUB_TYPE type)
    {
        m_vehicleSubType = type;
    }
    SetupWizard::VEHICLE_SUB_TYPE getVehicleSubType() const
    {
        return m_vehicleSubType;
    }

    void setInputType(Core::IBoardType::InputType type)
    {
        m_inputType = type;
    }
    Core::IBoardType::InputType getInputType() const
    {
        return m_inputType;
    }

    void setESCType(SetupWizard::ESC_TYPE type)
    {
        m_escType = type;
    }
    SetupWizard::ESC_TYPE getESCType() const
    {
        return m_escType;
    }

    void setGPSSetting(SetupWizard::GPS_SETTING setting)
    {
        m_gpsSetting = setting;
    }
    SetupWizard::GPS_SETTING getGPSSetting() const
    {
        return m_gpsSetting;
    }

    void setRadioSetting(SetupWizard::RADIO_SETTING setting)
    {
        m_radioSetting = setting;
    }
    SetupWizard::RADIO_SETTING getRadioSetting() const
    {
        return m_radioSetting;
    }

    void setLevellingBias(accelGyroBias bias)
    {
        m_calibrationBias = bias; m_calibrationPerformed = true;
    }
    bool isCalibrationPerformed() const
    {
        return m_calibrationPerformed;
    }
    accelGyroBias getCalibrationBias() const
    {
        return m_calibrationBias;
    }

    void setActuatorSettings(QList<actuatorChannelSettings> actuatorSettings)
    {
        m_actuatorSettings = actuatorSettings;
    }
    bool isMotorCalibrationPerformed() const
    {
        return m_motorCalibrationPerformed;
    }
    QList<actuatorChannelSettings> getActuatorSettings() const
    {
        return m_actuatorSettings;
    }

    void setRestartNeeded(bool needed)
    {
        m_restartNeeded = needed;
    }
    bool isRestartNeeded() const
    {
        return m_restartNeeded;
    }

    QString getSummaryText();

    Core::ConnectionManager *getConnectionManager()
    {
        if (!m_connectionManager) {
            m_connectionManager = Core::ICore::instance()->connectionManager();
            Q_ASSERT(m_connectionManager);
        }
        return m_connectionManager;
    }

private slots:
    void customBackClicked();
    void pageChanged(int currId);
private:
    enum { PAGE_START, PAGE_CONTROLLER, PAGE_VEHICLES, PAGE_MULTI, PAGE_FIXEDWING,
           PAGE_HELI, PAGE_SURFACE, PAGE_INPUT, PAGE_INPUT_NOT_SUPPORTED, PAGE_OUTPUT,
           PAGE_BIAS_CALIBRATION, PAGE_OUTPUT_CALIBRATION, PAGE_SAVE, PAGE_SUMMARY,
           PAGE_NOTYETIMPLEMENTED, PAGE_BOARDTYPE_UNKNOWN, PAGE_REBOOT, PAGE_END, PAGE_UPDATE };
    void createPages();
    bool saveHardwareSettings() const;
    bool canAutoUpdate() const;

    Core::IBoardType* m_controllerType;
    VEHICLE_TYPE m_vehicleType;
    VEHICLE_SUB_TYPE m_vehicleSubType;
    Core::IBoardType::InputType m_inputType;
    ESC_TYPE m_escType;

    GPS_SETTING m_gpsSetting;
    RADIO_SETTING m_radioSetting;

    bool m_calibrationPerformed;
    accelGyroBias m_calibrationBias;

    bool m_motorCalibrationPerformed;
    QList<actuatorChannelSettings> m_actuatorSettings;

    bool m_restartNeeded;

    bool m_back;

    Core::ConnectionManager *m_connectionManager;
};

#endif // SETUPWIZARD_H
