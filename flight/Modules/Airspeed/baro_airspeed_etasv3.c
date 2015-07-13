/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup AirspeedModule Airspeed Module
 * @{ 
 *
 * @file       baro_airspeed.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      Compute airspeed from multiple types of sensors
 *
 * @see        The GNU Public License (GPL) Version 3
 *
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

/**
 * Output object: BaroAirspeed
 *
 * This module will periodically update the value of the BaroAirspeed object.
 *
 */

#include "openpilot.h"
#include "airspeedsettings.h"
#include "baroairspeed.h"	// object that will be updated by the module
#include "pios_thread.h"

#if defined(PIOS_INCLUDE_ETASV3)

#define SAMPLING_DELAY_MS_ETASV3                50     //Update at 20Hz
#define ETS_AIRSPEED_SCALE                      1.0f   //???
#define CALIBRATION_IDLE_MS                     2000   //Time to wait before calibrating, in [ms]
#define CALIBRATION_COUNT_MS                    2000   //Time to spend calibrating, in [ms]
#define CALIBRATION_COUNT_IDLE                  (CALIBRATION_IDLE_MS/SAMPLING_DELAY_MS_ETASV3) //Ticks to wait before calibrating
#define CALIBRATION_COUNT                       (CALIBRATION_COUNT_MS/SAMPLING_DELAY_MS_ETASV3) //Ticks to wait while calibrating

// Private types

// Private variables
static uint16_t calibrationCount = 0;
static uint32_t calibrationSum = 0;

void baro_airspeedGetETASV3(BaroAirspeedData *baroAirspeedData, uint32_t *lastSysTime, uint8_t airspeedSensorType, int8_t airspeedADCPin)
{
	//Wait until our turn.
	PIOS_Thread_Sleep_Until(lastSysTime, SAMPLING_DELAY_MS_ETASV3);

	AirspeedSettingsData airspeedSettingsData;
	AirspeedSettingsGet(&airspeedSettingsData);

	//Check to see if airspeed sensor is returning baroAirspeedData
	baroAirspeedData->SensorValue = PIOS_ETASV3_ReadAirspeed();
	if (baroAirspeedData->SensorValue == -1) {
		baroAirspeedData->BaroConnected = BAROAIRSPEED_BAROCONNECTED_FALSE;
		baroAirspeedData->CalibratedAirspeed = 0;
		return;
	}
	
	//Calibrate sensor by averaging zero point value //THIS SHOULD NOT BE DONE IF THERE IS AN IN-AIR RESET. HOW TO DETECT THIS?
	if (calibrationCount < CALIBRATION_COUNT_IDLE) {
		calibrationCount++;
		calibrationSum = 0;
		return;
	} else if (calibrationCount < CALIBRATION_COUNT_IDLE + CALIBRATION_COUNT) {
		calibrationCount++;
		calibrationSum +=  baroAirspeedData->SensorValue;

		if (calibrationCount == CALIBRATION_COUNT_IDLE + CALIBRATION_COUNT) {
			airspeedSettingsData.ZeroPoint = (uint16_t) roundf(((float)calibrationSum) / CALIBRATION_COUNT);
			AirspeedSettingsZeroPointSet( &airspeedSettingsData.ZeroPoint );
		} else {
			return;
		}
	}

	//Compute airspeed
	float calibratedAirspeed;
	if (baroAirspeedData->SensorValue > airspeedSettingsData.ZeroPoint)
		calibratedAirspeed = ETS_AIRSPEED_SCALE * sqrtf(baroAirspeedData->SensorValue - airspeedSettingsData.ZeroPoint);
	else
		calibratedAirspeed = 0;

	baroAirspeedData->BaroConnected = BAROAIRSPEED_BAROCONNECTED_TRUE;
	baroAirspeedData->CalibratedAirspeed = calibratedAirspeed;
}

#else

void baro_airspeedGetETASV3(BaroAirspeedData *baroAirspeedData, uint32_t *lastSysTime, uint8_t airspeedSensorType, int8_t airspeedADCPin)
{
	/* Do nothing when driver support not compiled. */
	PIOS_Thread_Sleep_Until(lastSysTime, 1000);
}
#endif

/**
 * @}
 * @}
 */
