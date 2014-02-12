/**
 ******************************************************************************
 * @file       vehicletrim.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      Gui-less support class for vehicle trimming
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

#include "vehicletrim.h"
#include "configvehicletypewidget.h"

#include "actuatorcommand.h"
#include "actuatorsettings.h"
#include "stabilizationdesired.h"
#include "flightstatus.h"
#include "trimanglessettings.h"
#include "systemalarms.h"


VehicleTrim::VehicleTrim()
{
}

VehicleTrim::~VehicleTrim()
{
}


/**
 * @brief VehicleTrim::setFixedWingTrimAutopilotBias Takes the desired roll and pitch,
 * and sets that as the autopilot level bias.
 * @return success state
 */
VehicleTrim::autopilotLevelBiasMessages VehicleTrim::setAutopilotBias()
{
    SystemAlarms *systemAlarms = SystemAlarms::GetInstance(getObjectManager());
    FlightStatus *flightStatus = FlightStatus::GetInstance(getObjectManager());
    StabilizationDesired *stabilizationDesired = StabilizationDesired::GetInstance(getObjectManager());

    // Get TrimAnglesSettings UAVO
    TrimAnglesSettings *trimAnglesSettings = TrimAnglesSettings::GetInstance(getObjectManager());
    TrimAnglesSettings::DataFields trimAnglesSettingsData = trimAnglesSettings->getData();

    // Check that the receiver is present
    if (systemAlarms->getAlarm_ManualControl()  != SystemAlarms::ALARM_OK){
        return AUTOPILOT_LEVEL_FAILED_DUE_TO_MISSING_RECEIVER;
    }

    // Check that vehicle is disarmed
    if (flightStatus->getArmed() != FlightStatus::ARMED_DISARMED){
        return AUTOPILOT_LEVEL_FAILED_DUE_TO_ARMED_STATE;
    }

    // Check that vehicle is in stabilized{1,2,3} flight mode
    if (flightStatus->getFlightMode() != FlightStatus::FLIGHTMODE_STABILIZED1 &&
            flightStatus->getFlightMode() != FlightStatus::FLIGHTMODE_STABILIZED2 &&
            flightStatus->getFlightMode() != FlightStatus::FLIGHTMODE_STABILIZED3){
        return AUTOPILOT_LEVEL_FAILED_DUE_TO_FLIGHTMODE;
    }

    // Check that pitch and roll axes are in attitude mode
    if ((stabilizationDesired->getStabilizationMode_Pitch() != StabilizationDesired::STABILIZATIONMODE_ATTITUDE) ||
            (stabilizationDesired->getStabilizationMode_Roll() != StabilizationDesired::STABILIZATIONMODE_ATTITUDE)) {
        return AUTOPILOT_LEVEL_FAILED_DUE_TO_STABILIZATIONMODE;
    }

    // Increment the current pitch and roll settings by what the pilot is requesting
    trimAnglesSettingsData.Roll += stabilizationDesired->getRoll();
    trimAnglesSettingsData.Pitch += stabilizationDesired->getPitch();
    trimAnglesSettings->setData(trimAnglesSettingsData);
    trimAnglesSettings->updated();

    // Inform GUI that trim function has successfully completed
    emit trimCompleted();

    return AUTOPILOT_LEVEL_SUCCESS;
}


/**
 * @brief VehicleTrim::setFixedWingTrimActuators Reads the servo inputs from the transmitter, and sets
 * these values as the neutral points.
 * @return success state
 */
VehicleTrim::actuatorTrimMessages VehicleTrim::setTrimActuators()
{
    SystemAlarms *systemAlarms = SystemAlarms::GetInstance(getObjectManager());
    FlightStatus *flightStatus = FlightStatus::GetInstance(getObjectManager());

    // Get ActuatorCommand UAVO
    ActuatorCommand *actuatorCommand = ActuatorCommand::GetInstance(getObjectManager());
    ActuatorCommand::DataFields actuatorCommandData = actuatorCommand->getData();

    // Get ActuatorSettings UAVO
    ActuatorSettings *actuatorSettings = ActuatorSettings::GetInstance(getObjectManager());
    ActuatorSettings::DataFields actuatorSettingsData = actuatorSettings->getData();

    // Check that the receiver is present
    if (systemAlarms->getAlarm_ManualControl()  != SystemAlarms::ALARM_OK){
        return ACTUATOR_TRIM_FAILED_DUE_TO_MISSING_RECEIVER;
    }

    // Check that vehicle is in manual mode
    if (flightStatus->getFlightMode() != FlightStatus::FLIGHTMODE_MANUAL){
        return ACTUATOR_TRIM_FAILED_DUE_TO_FLIGHTMODE;
    }

    // Iterate over output channel descriptions
    QStringList channelDescriptions = ConfigVehicleTypeWidget::getChannelDescriptions();
    for(int i=0; i< channelDescriptions.length(); i++) {
        if(channelDescriptions[i] == "FixedWingRoll1" ||
                channelDescriptions[i] == "FixedWingRoll2")
        {
            int neutral = actuatorCommandData.Channel[i];
            actuatorSettingsData.ChannelNeutral[i] = neutral;
        }
        else if (channelDescriptions[i] == "FixedWingPitch1" ||
                 channelDescriptions[i] == "FixedWingPitch2")
        {
            int neutral = actuatorCommandData.Channel[i];
            actuatorSettingsData.ChannelNeutral[i] = neutral;
        }
        else if (channelDescriptions[i] == "FixedWingYaw1" ||
                 channelDescriptions[i] == "FixedWingYaw2")
        {
            int neutral = actuatorCommandData.Channel[i];
            actuatorSettingsData.ChannelNeutral[i] = neutral;
        }
    }

    // Set the data to the UAVO, and inform the flight controller that the UAVO has been updated
    actuatorSettings->setData(actuatorSettingsData);
    actuatorSettings->updated();

    // Inform GUI that trim function has successfully completed
    emit trimCompleted();

    return ACTUATOR_TRIM_SUCCESS;
}


/**
 * Util function to get a pointer to the object manager
 * @return pointer to the UAVObjectManager
 */
UAVObjectManager* VehicleTrim::getObjectManager() {
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objMngr);
    return objMngr;
}
