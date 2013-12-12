/**
 ******************************************************************************
 * @file       vehicletrim.h
 * @author     TauLabs, http://taulabs.org Copyright (C) 2013.
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
#ifndef VEHICLETRIM_H
#define VEHICLETRIM_H

#include <uavobjectmanager.h>
#include <extensionsystem/pluginmanager.h>
#include <uavobject.h>
#include <tempcompcurve.h>

#include <QObject>
#include <QTimer>
#include <QString>

/**
 * @brief The VehicleTrim class is a UI free algorithm that can be connected
 * to any interfaces.  As such it only communicates with the UI via signals
 * and slots, but has no direct handles to any particular controls or widgets.
 *
 */
class VehicleTrim : public QObject
{
    Q_OBJECT

public:
    explicit VehicleTrim();
    ~VehicleTrim();

    enum actuatorTrimMessages{
        ACTUATOR_TRIM_SUCCESS,
        ACTUATOR_TRIM_FAILED_DUE_TO_MISSING_RECEIVER,
        ACTUATOR_TRIM_FAILED_DUE_TO_FLIGHTMODE
    };

    enum autopilotLevelBiasMessages{
        AUTOPILOT_LEVEL_SUCCESS,
        AUTOPILOT_LEVEL_FAILED_DUE_TO_MISSING_RECEIVER,
        AUTOPILOT_LEVEL_FAILED_DUE_TO_ARMED_STATE,
        AUTOPILOT_LEVEL_FAILED_DUE_TO_FLIGHTMODE,
        AUTOPILOT_LEVEL_FAILED_DUE_TO_STABILIZATIONMODE
    };

    autopilotLevelBiasMessages setAutopilotBias();
    actuatorTrimMessages setTrimActuators();

private:

signals:

    //! Indicate that a trim process has successfully completed and the results saved to UAVO
    void trimCompleted();

private:

protected:

    //! Get the object manager
    UAVObjectManager* getObjectManager();

};

#endif // VEHICLETRIM_H
