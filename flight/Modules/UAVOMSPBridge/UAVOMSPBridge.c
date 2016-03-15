/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{
 * @addtogroup UAVOMSPBridge UAVO to MSP Bridge Module
 * @{
 *
 * @file       uavomspbridge.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015-2016
 * @author     dRonin, http://dronin.org Copyright (C) 2015-2016
 * @brief      Bridges selected UAVObjects to MSP for MWOSD and the like.
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

#include "openpilot.h"
#include "physical_constants.h"
#include "modulesettings.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "manualcontrolcommand.h"
#include "accessorydesired.h"
#include "attitudeactual.h"
#include "airspeedactual.h"
#include "actuatorsettings.h"
#include "actuatordesired.h"
#include "flightstatus.h"
#include "systemstats.h"
#include "systemalarms.h"
#include "homelocation.h"
#include "baroaltitude.h"
#include "pios_thread.h"
#include "pios_sensors.h"

#include "baroaltitude.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "modulesettings.h"

#include "msplib.h"

#if defined(PIOS_INCLUDE_MSP_BRIDGE)

#if defined(PIOS_MSP_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_MSP_STACK_SIZE
#else
#define STACK_SIZE_BYTES 768
#endif
#define TASK_PRIORITY               PIOS_THREAD_PRIO_LOW

#define MAX_ALARM_LEN 30

#define BOOT_DISPLAY_TIME_MS (10*1000)

static bool module_enabled;
extern uintptr_t pios_com_msp_id;
static struct msp_bridge *msp;
static int32_t uavoMSPBridgeInitialize(void);
static void uavoMSPBridgeTask(void *parameters);

static void msp_send_attitude(struct msp_bridge *m)
{
	union {
		uint8_t buf[0];
		struct {
			int16_t x;
			int16_t y;
			int16_t h;
		} att;
	} data;
	AttitudeActualData attActual;

	AttitudeActualGet(&attActual);

	// Roll and Pitch are in 10ths of a degree.
	data.att.x = attActual.Roll * 10;
	data.att.y = attActual.Pitch * -10;
	// Yaw is just -180 -> 180
	data.att.h = attActual.Yaw;

	msp_send_response(m, MSP_ATTITUDE, data.buf, sizeof(data));
}

static void msp_send_status(struct msp_bridge *m)
{
	union {
		uint8_t buf[0];
		struct {
			uint16_t cycleTime;
			uint16_t i2cErrors;
			uint16_t sensors;
			uint32_t flags;
			uint8_t setting;
		} __attribute__((packed)) status;
	} data;
	// TODO: https://github.com/TauLabs/TauLabs/blob/next/shared/uavobjectdefinition/actuatordesired.xml#L8
	data.status.cycleTime = 0;
	data.status.i2cErrors = 0;
	
	GPSPositionData gpsData;
	
	if (GPSPositionHandle() != NULL)
		GPSPositionGet(&gpsData);
	
	data.status.sensors = (PIOS_SENSORS_IsRegistered(PIOS_SENSOR_ACCEL) ? MSP_SENSOR_ACC  : 0) |
		(PIOS_SENSORS_IsRegistered(PIOS_SENSOR_BARO) ? MSP_SENSOR_BARO : 0) |
		(PIOS_SENSORS_IsRegistered(PIOS_SENSOR_MAG) ? MSP_SENSOR_MAG : 0) |
		(gpsData.Status != GPSPOSITION_STATUS_NOGPS ? MSP_SENSOR_GPS : 0);
	
	data.status.flags = 0;
	data.status.setting = 0;

	if (FlightStatusHandle() != NULL) {
		FlightStatusData flight_status;
		FlightStatusGet(&flight_status);

		data.status.flags = flight_status.Armed == FLIGHTSTATUS_ARMED_ARMED;

		for (int i = 1; msp_boxes[i].mode != MSP_BOX_LAST; i++) {
			if (flight_status.FlightMode == msp_boxes[i].tlmode) {
				data.status.flags |= (1 << i);
			}
		}
	}

	msp_send_response(m, MSP_STATUS, data.buf, sizeof(data));
}

static void msp_send_analog(struct msp_bridge *m)
{
	union {
		uint8_t buf[0];
		struct {
			uint8_t vbat;
			uint16_t powerMeterSum;
			uint16_t rssi;
			uint16_t current;
		} __attribute__((packed)) status;
	} data;
	data.status.powerMeterSum = 0;

	FlightBatterySettingsData batSettings;
	FlightBatteryStateData batState;

	if (FlightBatteryStateHandle() != NULL)
		FlightBatteryStateGet(&batState);
	if (FlightBatterySettingsHandle() != NULL) {
		FlightBatterySettingsGet(&batSettings);
	}

	if (batSettings.VoltagePin != FLIGHTBATTERYSETTINGS_VOLTAGEPIN_NONE)
		data.status.vbat = (uint8_t)lroundf(batState.Voltage * 10);

	if (batSettings.CurrentPin != FLIGHTBATTERYSETTINGS_CURRENTPIN_NONE) {
		data.status.current = lroundf(batState.Current * 100);
		data.status.powerMeterSum = lroundf(batState.ConsumedEnergy);
	}

	ManualControlCommandData manualState;
	ManualControlCommandGet(&manualState);

	// MSP RSSI's range is 0-1023
	data.status.rssi = (manualState.Rssi >= 0 && manualState.Rssi <= 100) ? manualState.Rssi * 10 : 0;

	msp_send_response(m, MSP_ANALOG, data.buf, sizeof(data));
}

static void msp_send_ident(struct msp_bridge *m)
{
	// TODO
}

static void msp_send_raw_gps(struct msp_bridge *m)
{
	union {
		uint8_t buf[0];
		struct {
			uint8_t fix;                 // 0 or 1
			uint8_t num_sat;
			int32_t lat;                  // 1 / 10 000 000 deg
			int32_t lon;                  // 1 / 10 000 000 deg
			uint16_t alt;                 // meter
			uint16_t speed;               // cm/s
			int16_t ground_course;        // degree * 10
		} __attribute__((packed)) raw_gps;
	} data;
	
	GPSPositionData gps_data;
	
	if (GPSPositionHandle() != NULL) {
		GPSPositionGet(&gps_data);
		data.raw_gps.fix           = (gps_data.Status >= GPSPOSITION_STATUS_FIX2D ? 1 : 0);  // Data will display on OSD if 2D fix or better
		data.raw_gps.num_sat       = gps_data.Satellites;
		data.raw_gps.lat           = gps_data.Latitude;
		data.raw_gps.lon           = gps_data.Longitude;
		data.raw_gps.alt           = gps_data.Altitude;
		data.raw_gps.speed         = gps_data.Groundspeed;
		data.raw_gps.ground_course = gps_data.Heading * 10;
	} else {
		data.raw_gps.fix           = 0;  // Data won't display on OSD
		data.raw_gps.num_sat       = 0;
		data.raw_gps.lat           = 0;
		data.raw_gps.lon           = 0;
		data.raw_gps.alt           = 0;
		data.raw_gps.speed         = 0;
		data.raw_gps.ground_course = 0;
	}
	
	msp_send_response(m, MSP_RAW_GPS, data.buf, sizeof(data));
}

static void msp_send_comp_gps(struct msp_bridge *m)
{
	union {
		uint8_t buf[0];
		struct {
			uint16_t distance_to_home;     // meter
			int16_t  direction_to_home;    // degree [-180:180]
			uint8_t  home_position_valid;  // 0 = Invalid
		} __attribute__((packed)) comp_gps;
	} data;
	
	GPSPositionData gps_data;
	HomeLocationData home_data;
	
	if ((GPSPositionHandle() == NULL) || (HomeLocationHandle() == NULL)) {
		data.comp_gps.distance_to_home    = 0;
		data.comp_gps.direction_to_home   = 0;
		data.comp_gps.home_position_valid = 0;  // Home distance and direction will not display on OSD
	} else {
		GPSPositionGet(&gps_data);
		HomeLocationGet(&home_data);
		
		if((gps_data.Status < GPSPOSITION_STATUS_FIX2D) || (home_data.Set == HOMELOCATION_SET_FALSE)) {
			data.comp_gps.distance_to_home    = 0;
			data.comp_gps.direction_to_home   = 0;
			data.comp_gps.home_position_valid = 0;  // Home distance and direction will not display on OSD
		} else {
			data.comp_gps.home_position_valid = 1;  // Home distance and direction will display on OSD
			
			int32_t delta_lon = (home_data.Longitude - gps_data.Longitude);  // degrees * 1e7
			int32_t delta_lat = (home_data.Latitude  - gps_data.Latitude );  // degrees * 1e7
	
			float delta_y = delta_lon * WGS84_RADIUS_EARTH_KM * DEG2RAD;  // KM * 1e7
			float delta_x = delta_lat * WGS84_RADIUS_EARTH_KM * DEG2RAD;  // KM * 1e7
	
			delta_y *= cosf(home_data.Latitude * 1e-7f * (float)DEG2RAD);  // Latitude compression correction
	
			data.comp_gps.distance_to_home  = (uint16_t)(sqrtf(delta_x * delta_x + delta_y * delta_y) * 1e-4f);  // meters
	
			if ((delta_lon == 0) && (delta_lat == 0))
				data.comp_gps.direction_to_home = 0;
			else
				data.comp_gps.direction_to_home = (int16_t)(atan2f(delta_y, delta_x) * RAD2DEG + 0.5f); // degrees;
		}
	}

	msp_send_response(m, MSP_COMP_GPS, data.buf, sizeof(data));
}

static void msp_send_altitude(struct msp_bridge *m)
{
	union {
		uint8_t buf[0];
		struct {
			int32_t alt; // cm
			uint16_t vario; // cm/s
		} __attribute__((packed)) baro;
	} data;

	BaroAltitudeData baro;
	if (BaroAltitudeHandle() != NULL)
		BaroAltitudeGet(&baro);

	data.baro.alt = (int32_t)roundf(baro.Altitude * 100.0f);

	msp_send_response(m, MSP_ALTITUDE, data.buf, sizeof(data));
}

// Scale stick values whose input range is -1 to 1 to MSP's expected
// PWM range of 1000-2000.
static uint16_t msp_scale_rc(float percent) {
	return percent*500 + 1500;
}

// Throttle maps to 1100-1900 and a bit differently as -1 == 1000 and
// then jumps immediately to 0 -> 1 for the rest of the range.  MWOSD
// can learn ranges as wide as they are sent, but defaults to
// 1100-1900 so the throttle indicator will vary for the same stick
// position unless we send it what it wants right away.
static uint16_t msp_scale_rc_thr(float percent) {
	return percent <= 0 ? 1100 : percent*800 + 1100;
}

// MSP RC order is Roll/Pitch/Yaw/Throttle/AUX1/AUX2/AUX3/AUX4
static void msp_send_channels(struct msp_bridge *m)
{
	AccessoryDesiredData acc0, acc1, acc2;
	ManualControlCommandData manualState;
	ManualControlCommandGet(&manualState);
	AccessoryDesiredInstGet(0, &acc0);
	AccessoryDesiredInstGet(1, &acc1);
	AccessoryDesiredInstGet(2, &acc2);

	union {
		uint8_t buf[0];
		uint16_t channels[8];
	} data = {
		.channels = {
			msp_scale_rc(manualState.Roll),
			msp_scale_rc(manualState.Pitch * -1), // TL pitch is backwards
			msp_scale_rc(manualState.Yaw),
			msp_scale_rc_thr(manualState.Throttle),
			msp_scale_rc(acc0.AccessoryVal),
			msp_scale_rc(acc1.AccessoryVal),
			msp_scale_rc(acc2.AccessoryVal),
			1000, // no aux4
		}
	};

	msp_send_response(m, MSP_RC, data.buf, sizeof(data));
}

static void msp_send_boxids(struct msp_bridge *m) {
	uint8_t boxes[MSP_BOX_LAST];
	int len = 0;

	for (int i = 0; msp_boxes[i].mode != MSP_BOX_LAST; i++) {
		boxes[len++] = msp_boxes[i].mwboxid;
	}
	msp_send_response(m, MSP_BOXIDS, boxes, len);
}

#define ALARM_OK 0
#define ALARM_WARN 1
#define ALARM_ERROR 2
#define ALARM_CRIT 3

static void msp_send_alarms(struct msp_bridge *m) {
	union {
		uint8_t buf[0];
		struct {
			uint8_t state;
			char msg[MAX_ALARM_LEN];
		} __attribute__((packed)) alarm;
	} data;

	SystemAlarmsData alarm;
	SystemAlarmsGet(&alarm);

	// Special case early boot times -- just report boot reason
	if (PIOS_Thread_Systime() < BOOT_DISPLAY_TIME_MS) {
		data.alarm.state = ALARM_CRIT;
		const char *boot_reason = AlarmBootReason(alarm.RebootCause);
		strncpy((char*)data.alarm.msg, boot_reason, MAX_ALARM_LEN);
		data.alarm.msg[MAX_ALARM_LEN-1] = '\0';
		msp_send_response(m, MSP_ALARMS, data.buf, strlen((char*)data.alarm.msg)+1);
		return;
	}

	uint8_t state;
	data.alarm.state = ALARM_OK;
	int32_t len = AlarmString(&alarm, data.alarm.msg,
				  sizeof(data.alarm.msg), false, &state);
	switch (state) {
	case SYSTEMALARMS_ALARM_WARNING:
		data.alarm.state = ALARM_WARN;
		break;
	case SYSTEMALARMS_ALARM_ERROR:
		break;
	case SYSTEMALARMS_ALARM_CRITICAL:
		data.alarm.state = ALARM_CRIT;;
	}

	msp_send_response(m, MSP_ALARMS, data.buf, len+1);
}

static bool msp_response_cb(struct msp_bridge *m, uint8_t cmd, const uint8_t *data, size_t len)
{
	// Respond to interesting things.
	switch (cmd) {
	case MSP_IDENT:
		msp_send_ident(m);
		return true;
	case MSP_RAW_GPS:
		msp_send_raw_gps(m);
		return true;
	case MSP_COMP_GPS:
		msp_send_comp_gps(m);
		return true;
	case MSP_ALTITUDE:
		msp_send_altitude(m);
		return true;
	case MSP_ATTITUDE:
		msp_send_attitude(m);
		return true;
	case MSP_STATUS:
		msp_send_status(m);
		return true;
	case MSP_ANALOG:
		msp_send_analog(m);
		return true;
	case MSP_RC:
		msp_send_channels(m);
		return true;
	case MSP_BOXIDS:
		msp_send_boxids(m);
		return true;
	case MSP_ALARMS:
		msp_send_alarms(m);
		return true;
	}
	return false;
}

/**
 * Module start routine automatically called after initialization routine
 * @return 0 when was successful
 */
static int32_t uavoMSPBridgeStart(void)
{
	if (!module_enabled) {
		// give port to telemetry if it doesn't have one
		// stops board getting stuck in condition where it can't be connected to gcs
		if(!PIOS_COM_TELEM_RF)
			PIOS_COM_TELEM_RF = pios_com_msp_id;

		return -1;
	}

	struct pios_thread *task = PIOS_Thread_Create(
		uavoMSPBridgeTask, "uavoMSPBridge",
		STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	TaskMonitorAdd(TASKINFO_RUNNING_UAVOMSPBRIDGE,
			task);

	return 0;
}

static void setMSPSpeed(struct msp_bridge *m)
{
	if (m->com) {
		uint8_t speed;
		ModuleSettingsMSPSpeedGet(&speed);

		switch (speed) {
		case MODULESETTINGS_MSPSPEED_2400:
			PIOS_COM_ChangeBaud(m->com, 2400);
			break;
		case MODULESETTINGS_MSPSPEED_4800:
			PIOS_COM_ChangeBaud(m->com, 4800);
			break;
		case MODULESETTINGS_MSPSPEED_9600:
			PIOS_COM_ChangeBaud(m->com, 9600);
			break;
		case MODULESETTINGS_MSPSPEED_19200:
			PIOS_COM_ChangeBaud(m->com, 19200);
			break;
		case MODULESETTINGS_MSPSPEED_38400:
			PIOS_COM_ChangeBaud(m->com, 38400);
			break;
		case MODULESETTINGS_MSPSPEED_57600:
			PIOS_COM_ChangeBaud(m->com, 57600);
			break;
		case MODULESETTINGS_MSPSPEED_115200:
			PIOS_COM_ChangeBaud(m->com, 115200);
			break;
		}
	}
}

/**
 * Module initialization routine
 * @return 0 when initialization was successful
 */
static int32_t uavoMSPBridgeInitialize(void)
{
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);

	if (pios_com_msp_id && (module_state[MODULESETTINGS_ADMINSTATE_UAVOMSPBRIDGE]
			== MODULESETTINGS_ADMINSTATE_ENABLED)) {

		msp = msp_init(pios_com_msp_id);
		if (msp != NULL) {
			setMSPSpeed(msp);
			msp_set_request_cb(msp, msp_response_cb);

			module_enabled = true;

			return 0;
		}

	}

	module_enabled = false;

	return -1;
}
MODULE_INITCALL(uavoMSPBridgeInitialize, uavoMSPBridgeStart)

/**
 * Main task routine
 * @param[in] parameters parameter given by PIOS_Thread_Create()
 */
static void uavoMSPBridgeTask(void *parameters)
{
	while (1) {
		uint8_t b = 0;
		uint16_t count = PIOS_COM_ReceiveBuffer(msp->com, &b, 1, PIOS_QUEUE_TIMEOUT_MAX);
		if (count) {
			if (!msp_receive_byte(msp, b)) {
				// Returning is considered risky here as
				// that's unusual and this is an edge case.
				while (1) {
					PIOS_Thread_Sleep(60*1000);
				}
			}
		}
	}
}

#endif //PIOS_INCLUDE_MSP_BRIDGE
/**
 * @}
 * @}
 */
