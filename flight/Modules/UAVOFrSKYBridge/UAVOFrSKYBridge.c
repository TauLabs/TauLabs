/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup UAVOFrSKYBridge UAVO to FrSKY Bridge Module
 * @{ 
 *
 * @file       UAVOFrSKYBridge.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Bridges selected UAVObjects to Mavlink
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

// ****************
#include "openpilot.h"
#include "physical_constants.h"
#include "modulesettings.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "manualcontrolcommand.h"
#include "attitudeactual.h"
#include "airspeedactual.h"
#include "actuatordesired.h"
#include "flightstatus.h"
#include "systemstats.h"
#include "homelocation.h"
#include "baroaltitude.h"
#include "mavlink.h"

// ****************
// Private functions

static void uavoFrSKYBridgeTask(void *parameters);
static uint16_t frsky_pack_frame_01(
		float accels_x,
		float accels_y,
		float accels_z,
		float altitude;
		float temperature_01,
		float temperature_02,
		float voltage,
		float current,
		uint16_t RPM;
		uint8_t *buf);

// ****************
// Private constants

#if defined(PIOS_FRSKY_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_FRSKY_STACK_SIZE
#else
#define STACK_SIZE_BYTES 800
#endif

#define TASK_PRIORITY               (tskIDLE_PRIORITY + 1)
#define TASK_RATE_HZ				10

#define FRSKY_MAX_PACKET_LEN 53

enum FRSKY_FRAME
{
    FRSKY_FRAME_01,
    FRSKY_FRAME_02,
    FRSKY_FRAME_03,
    FRSKY_FRAME_COUNT
};

static const uint8_t frsky_rates[] =
	 { [FRSKY_FRAME_01]=0x05, //5Hz
	   [FRSKY_FRAME_02]=0x01, //1Hz
	   [FRSKY_FRAME_03]=0x01 }; //2Hz

#define MAXSTREAMS sizeof(frsky_rates)

// ****************
// Private variables

static xTaskHandle uavoFrSKYBridgeTaskHandle;

static uint32_t frsky_port;

static bool module_enabled = false;

static uint8_t * frame_ticks;

static uint8_t * serial_buf;

/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t uavoFrSKYBridgeStart(void) {
	if (module_enabled) {
		// Start tasks
		xTaskCreate(uavoFrSKYBridgeTask, (signed char *) "uavoFrSkyBridge",
				STACK_SIZE_BYTES / 4, NULL, TASK_PRIORITY,
				&uavoFrSKYBridgeTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_UAVOFRSKYBRIDGE,
				uavoFrSKYBridgeTaskHandle);
		return 0;
	}
	return -1;
}
/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t uavoFrSKYInitialize(void) {
	frsky_port = PIOS_COM_FRSKY;

	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);

	if (frsky_port
			&& (module_state[MODULESETTINGS_ADMINSTATE_UAVOFRSKYBRIDGE]
					== MODULESETTINGS_ADMINSTATE_ENABLED)) {
		module_enabled = true;
		PIOS_COM_ChangeBaud(frsky_port, 9600);

		serial_buf = pvPortMalloc(FRSKY_MAX_PACKET_LEN);
		stream_ticks = pvPortMalloc(MAXSTREAMS);
		for (int x = 0; x < MAXSTREAMS; ++x) {
			stream_ticks[x] = (TASK_RATE_HZ / mav_rates[x]);
		}
	} else {
		module_enabled = false;
	}
	return 0;
}
MODULE_INITCALL( uavoFrSKYInitialize, uavoFrSKYStart)

/**
 * Main task. It does not return.
 */

static void uavoFrSKYBridgeTask(void *parameters) {
	FlightBatterySettingsData batSettings;
	FlightBatteryStateData batState;
	GPSPositionData gpsPosData;
	BaroAltitudeData baroAltitude;
	AccelsData accels;

	if (FlightBatterySettingsHandle() != NULL )
		FlightBatterySettingsGet(&batSettings);
	else {
		batSettings.Capacity = 0;
		batSettings.NbCells = 0;
		batSettings.SensorCalibrationFactor[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONFACTOR_CURRENT] = 0;
		batSettings.SensorCalibrationFactor[FLIGHTBATTERYSETTINGS_SENSORCALIBRATIONFACTOR_VOLTAGE] = 0;
		batSettings.SensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYCURRENT] = FLIGHTBATTERYSETTINGS_SENSORTYPE_DISABLED;
		batSettings.SensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYVOLTAGE] = FLIGHTBATTERYSETTINGS_SENSORTYPE_DISABLED;
		batSettings.Type = FLIGHTBATTERYSETTINGS_TYPE_NONE;
		batSettings.VoltageThresholds[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_WARNING] = 0;
		batSettings.VoltageThresholds[FLIGHTBATTERYSETTINGS_VOLTAGETHRESHOLDS_ALARM] = 0;
	}

	if (GPSPositionHandle() == NULL ){
		gpsPosData.Altitude = 0;
		gpsPosData.GeoidSeparation = 0;
		gpsPosData.Groundspeed = 0;
		gpsPosData.HDOP = 0;
		gpsPosData.Heading = 0;
		gpsPosData.Latitude = 0;
		gpsPosData.Longitude = 0;
		gpsPosData.PDOP = 0;
		gpsPosData.Satellites = 0;
		gpsPosData.Status = 0;
		gpsPosData.VDOP = 0;
	}

	if (FlightBatteryStateHandle() == NULL ) {
		batState.AvgCurrent = 0;
		batState.BoardSupplyVoltage = 0;
		batState.ConsumedEnergy = 0;
		batState.Current = 0;
		batState.EstimatedFlightTime = 0;
		batState.PeakCurrent = 0;
		batState.Voltage = 0;
	}

	if (AccelsHandle() != NULL )
		AccelsGet(&accels);
	else {
		accels.x = 0.0;
		accels.y = 0.0;
		accels.z = 0.0;
		accels.temperature = 0.0;
	}

	uint16_t msg_length;
	portTickType lastSysTime;
	// Main task loop
	lastSysTime = xTaskGetTickCount();

	while (1) {
		vTaskDelayUntil(&lastSysTime, MS2TICKS(1000 / TASK_RATE_HZ));

		if (stream_trigger(FRSKY_FRAME_01)) {
			if (FlightBatteryStateHandle() != NULL )
				FlightBatteryStateGet(&batState);

			if (AccelsHandle() != NULL )
				AccelsGet(&accels);

			if (BaroAltitudeHandle() != NULL )
				BaroAltitudeGet(&baroAltitude);

			float voltage = 0.0;
			if (batSettings.SensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYVOLTAGE] == FLIGHTBATTERYSETTINGS_SENSORTYPE_ENABLED)
				voltage = batState.Voltage * 1000;

			float current = 0.0;
			if (batSettings.SensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYCURRENT] == FLIGHTBATTERYSETTINGS_SENSORTYPE_ENABLED)
				current = batState.Current * 100;

			msg_length = uavoFrSKYBridgePackFrame_01(
					accels.x,
					accels.y,
					accels.z,
					baroAltitude.Altitude,
					baroAltitude.Temperature,
					0.0,
					voltage,
					current,
					0,
					serial_buf);

			PIOS_COM_SendBuffer(frsky_port, serial_buf, msg_length);
		}

		if (stream_trigger(FRSKY_FRAME_02)) {
		}

		if (stream_trigger(FRSKY_FRAME_03)) {
		}
	}
}

static bool frame_trigger(enum FRSKY_FRAME frame_num) {
	uint8_t rate = (uint8_t) frsky_rates[frame_num];

	if (rate == 0) {
		return false;
	}

	if (frame_ticks[frame_num] == 0) {
		// we're triggering now, setup the next trigger point
		if (rate > TASK_RATE_HZ) {
			rate = TASK_RATE_HZ;
		}
		frame_ticks[frame_num] = (TASK_RATE_HZ / rate);
		return true;
	}

	// count down at 50Hz
	frame_ticks[frame_num]--;
	return false;
}

static uint16_t frsky_pack_frame_01(
		float accels_x,
		float accels_y,
		float accels_z,
		float altitude;
		float temperature_01,
		float temperature_02,
		float voltage,
		float current,
		uint16_t RPM;
		uint8_t *buf)
{
	return 0;
}
/**
 * @}
 * @}
 */
