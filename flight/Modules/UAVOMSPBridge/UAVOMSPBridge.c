/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{
 * @addtogroup UAVOMSPBridge UAVO to MSP Bridge Module
 * @{
 *
 * @file       uavomspbridge.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
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

#if defined(PIOS_INCLUDE_MSP_BRIDGE)

#define MSP_SENSOR_ACC 1
#define MSP_SENSOR_BARO 2
#define MSP_SENSOR_MAG 4
#define MSP_SENSOR_GPS 8

// Magic numbers copied from mwosd
#define  MSP_IDENT      100 // multitype + multiwii version + protocol version + capability variable
#define  MSP_STATUS     101 // cycletime & errors_count & sensor present & box activation & current setting number
#define  MSP_RAW_IMU    102 // 9 DOF
#define  MSP_SERVO      103 // 8 servos
#define  MSP_MOTOR      104 // 8 motors
#define  MSP_RC         105 // 8 rc chan and more
#define  MSP_RAW_GPS    106 // fix, numsat, lat, lon, alt, speed, ground course
#define  MSP_COMP_GPS   107 // distance home, direction home
#define  MSP_ATTITUDE   108 // 2 angles 1 heading
#define  MSP_ALTITUDE   109 // altitude, variometer
#define  MSP_ANALOG     110 // vbat, powermetersum, rssi if available on RX
#define  MSP_RC_TUNING  111 // rc rate, rc expo, rollpitch rate, yaw rate, dyn throttle PID
#define  MSP_PID        112 // P I D coeff (9 are used currently)
#define  MSP_BOX        113 // BOX setup (number is dependant of your setup)
#define  MSP_MISC       114 // powermeter trig
#define  MSP_MOTOR_PINS 115 // which pins are in use for motors & servos, for GUI
#define  MSP_BOXNAMES   116 // the aux switch names
#define  MSP_PIDNAMES   117 // the PID names
#define  MSP_BOXIDS     119 // get the permanent IDs associated to BOXes
#define  MSP_NAV_STATUS 121 // Returns navigation status
#define  MSP_CELLS      130 // FrSky SPort Telemtry
#define  MSP_ALARMS     242 // Alarm request

typedef enum {
	MSP_BOX_ARM,
	MSP_BOX_ANGLE,
	MSP_BOX_HORIZON,
	MSP_BOX_BARO,
	MSP_BOX_VARIO,
	MSP_BOX_MAG,
	MSP_BOX_GPSHOME,
	MSP_BOX_GPSHOLD,
	MSP_BOX_LAST,
} msp_box_t;

static struct {
	msp_box_t mode;
	uint8_t mwboxid;
	FlightStatusFlightModeOptions tlmode;
} msp_boxes[] = {
	{ MSP_BOX_ARM, 0, 0 },
	{ MSP_BOX_ANGLE, 1, FLIGHTSTATUS_FLIGHTMODE_LEVELING},
	{ MSP_BOX_HORIZON, 2, FLIGHTSTATUS_FLIGHTMODE_HORIZON},
	{ MSP_BOX_BARO, 3, FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD},
	{ MSP_BOX_VARIO, 4, 0},
	{ MSP_BOX_MAG, 5, 0},
	{ MSP_BOX_GPSHOME, 10, FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME},
	{ MSP_BOX_GPSHOLD, 11, FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD},
	{ MSP_BOX_LAST, 0xff, 0},
};

typedef enum {
	MSP_IDLE,
	MSP_HEADER_START,
	MSP_HEADER_M,
	MSP_HEADER_SIZE,
	MSP_HEADER_CMD,
	MSP_FILLBUF,
	MSP_CHECKSUM,
	MSP_DISCARD,
	MSP_MAYBE_UAVTALK2,
	MSP_MAYBE_UAVTALK3,
	MSP_MAYBE_UAVTALK4,
} msp_state;

struct msp_bridge {
	uintptr_t com;

	msp_state state;
	uint8_t cmd_size;
	uint8_t cmd_id;
	uint8_t cmd_i;
	uint8_t checksum;
	union {
		uint8_t data[0];
		// Specific packed data structures go here.
	} cmd_data;
};

#if defined(PIOS_MSP_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_MSP_STACK_SIZE
#else
#define STACK_SIZE_BYTES 672
#endif
#define TASK_PRIORITY               PIOS_THREAD_PRIO_LOW

#define MAX_ALARM_LEN 30

#define BOOT_DISPLAY_TIME_MS (10*1000)

static bool module_enabled;
extern uintptr_t pios_com_msp_id;
static struct msp_bridge *msp;
static int32_t uavoMSPBridgeInitialize(void);
static void uavoMSPBridgeTask(void *parameters);

static void msp_send(struct msp_bridge *m, uint8_t cmd, const uint8_t *data, size_t len)
{
	uint8_t buf[5];
	uint8_t cs = (uint8_t)(len) ^ cmd;

	buf[0] = '$';
	buf[1] = 'M';
	buf[2] = '>';
	buf[3] = (uint8_t)(len);
	buf[4] = cmd;

	PIOS_COM_SendBuffer(m->com, buf, sizeof(buf));
	PIOS_COM_SendBuffer(m->com, data, len);

	for (int i = 0; i < len; i++) {
		cs ^= data[i];
	}
	cs ^= 0;

	buf[0] = cs;
	PIOS_COM_SendBuffer(m->com, buf, 1);
}

static msp_state msp_state_size(struct msp_bridge *m, uint8_t b)
{
	m->cmd_size = b;
	m->checksum = b;
	return MSP_HEADER_CMD;
}

static msp_state msp_state_cmd(struct msp_bridge *m, uint8_t b)
{
	m->cmd_i = 0;
	m->cmd_id = b;
	m->checksum ^= m->cmd_id;

	if (m->cmd_size > sizeof(m->cmd_data)) {
		// Too large a body.  Let's ignore it.
		return MSP_DISCARD;
	}

	return m->cmd_size == 0 ? MSP_CHECKSUM : MSP_FILLBUF;
}

static msp_state msp_state_fill_buf(struct msp_bridge *m, uint8_t b)
{
	m->cmd_data.data[m->cmd_i++] = b;
	m->checksum ^= b;
	return m->cmd_i == m->cmd_size ? MSP_CHECKSUM : MSP_FILLBUF;
}

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

	msp_send(m, MSP_ATTITUDE, data.buf, sizeof(data));
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
	data.status.sensors =
		(PIOS_SENSORS_IsRegistered(PIOS_SENSOR_ACCEL) ? MSP_SENSOR_ACC : 0)
		| (PIOS_SENSORS_IsRegistered(PIOS_SENSOR_BARO) ? MSP_SENSOR_BARO : 0)
		| (PIOS_SENSORS_IsRegistered(PIOS_SENSOR_MAG) ? MSP_SENSOR_MAG : 0);
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

	msp_send(m, MSP_STATUS, data.buf, sizeof(data));
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

	FlightBatterySettingsData batSettings = {};
	FlightBatteryStateData batState = {};

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

	msp_send(m, MSP_ANALOG, data.buf, sizeof(data));
}

static void msp_send_ident(struct msp_bridge *m)
{
	// TODO
}

static void msp_send_raw_gps(struct msp_bridge *m)
{
	// TODO
}

static void msp_send_comp_gps(struct msp_bridge *m)
{
	// TODO
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

	msp_send(m, MSP_ALTITUDE, data.buf, sizeof(data));
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

	msp_send(m, MSP_RC, data.buf, sizeof(data));
}

static void msp_send_boxids(struct msp_bridge *m) {
	uint8_t boxes[MSP_BOX_LAST];
	int len = 0;

	for (int i = 0; msp_boxes[i].mode != MSP_BOX_LAST; i++) {
		boxes[len++] = msp_boxes[i].mwboxid;
	}
	msp_send(m, MSP_BOXIDS, boxes, len);
}

static const char alarm_names[][8] = {
	[SYSTEMALARMS_ALARM_OUTOFMEMORY] = "MEMORY",
	[SYSTEMALARMS_ALARM_CPUOVERLOAD] = "CPU",
	[SYSTEMALARMS_ALARM_STACKOVERFLOW] = "STACK",
	[SYSTEMALARMS_ALARM_SYSTEMCONFIGURATION] = "CONFIG",
	[SYSTEMALARMS_ALARM_EVENTSYSTEM] = "EVENT",
	[SYSTEMALARMS_ALARM_TELEMETRY] = {0}, // ignored
	[SYSTEMALARMS_ALARM_MANUALCONTROL] = "MANUAL",
	[SYSTEMALARMS_ALARM_ACTUATOR] = "ACTUATOR",
	[SYSTEMALARMS_ALARM_ATTITUDE] = "ATTITUDE",
	[SYSTEMALARMS_ALARM_SENSORS] = "SENSORS",
	[SYSTEMALARMS_ALARM_STABILIZATION] = "STAB",
	[SYSTEMALARMS_ALARM_PATHFOLLOWER] = "PATH-F",
	[SYSTEMALARMS_ALARM_PATHPLANNER] = "PATH-P",
	[SYSTEMALARMS_ALARM_BATTERY] = "BATTERY",
	[SYSTEMALARMS_ALARM_FLIGHTTIME] = "TIME",
	[SYSTEMALARMS_ALARM_I2C] = "I2C",
	[SYSTEMALARMS_ALARM_GPS] = "GPS",
	[SYSTEMALARMS_ALARM_ALTITUDEHOLD] = "A-HOLD",
	[SYSTEMALARMS_ALARM_BOOTFAULT] = "BOOT",
	[SYSTEMALARMS_ALARM_GEOFENCE] = "GEOFENCE",
	[SYSTEMALARMS_ALARM_TEMPBARO] = "TEMPBARO",
	[SYSTEMALARMS_ALARM_GYROBIAS] = "GYROBIAS",
	[SYSTEMALARMS_ALARM_ADC] = "ADC",
};

// If someone adds a new alarm, we'd like it added to the array above.
DONT_BUILD_IF(NELEMENTS(alarm_names) != SYSTEMALARMS_ALARM_NUMELEM, AlarmArrayMismatch);
DONT_BUILD_IF(sizeof(*alarm_names) >= MAX_ALARM_LEN, BufferOverflow);

static const char config_error_names[][14] = {
	[SYSTEMALARMS_CONFIGERROR_STABILIZATION] = "CFG:STAB",
	[SYSTEMALARMS_CONFIGERROR_MULTIROTOR] = "CFG:MULTIROTOR",
	[SYSTEMALARMS_CONFIGERROR_AUTOTUNE] = "CFG:AUTOTUNE",
	[SYSTEMALARMS_CONFIGERROR_ALTITUDEHOLD] = "CFG:AH1",
	[SYSTEMALARMS_CONFIGERROR_POSITIONHOLD] = "CFG:POS-HOLD",
	[SYSTEMALARMS_CONFIGERROR_PATHPLANNER] = "CFG:PATHPLAN",
	[SYSTEMALARMS_CONFIGERROR_DUPLICATEPORTCFG] = "CFG:DUP PORT",
	[SYSTEMALARMS_CONFIGERROR_NAVFILTER] = "CFG:NAVFILTER",
	[SYSTEMALARMS_CONFIGERROR_UNSAFETOARM] = "CFG:UNSAFE",
	[SYSTEMALARMS_CONFIGERROR_UNDEFINED] = "CFG:UNDEF",
	[SYSTEMALARMS_CONFIGERROR_NONE] = {0},
};

// DONT_BUILD_IF(NELEMENTS(CONFIG_ERROR_NAMES) != SYSTEMALARMS_CONFIGERROR_NUMELEM, AlarmArrayMismatch);
DONT_BUILD_IF(sizeof(*config_error_names) >= MAX_ALARM_LEN, BufferOverflow);

static const char manual_control_names[][12] = {
	[SYSTEMALARMS_MANUALCONTROL_SETTINGS] = "MAN:SETTINGS",
	[SYSTEMALARMS_MANUALCONTROL_NORX] = "MAN:NO RX",
	[SYSTEMALARMS_MANUALCONTROL_ACCESSORY] = "MAN:ACC",
	[SYSTEMALARMS_MANUALCONTROL_ALTITUDEHOLD] = "MAN:A-HOLD",
	[SYSTEMALARMS_MANUALCONTROL_PATHFOLLOWER] = "MAN:PATH-F",
	[SYSTEMALARMS_MANUALCONTROL_UNDEFINED] = "MAN:UNDEF",
	[SYSTEMALARMS_MANUALCONTROL_NONE] = {0},
};

// DONT_BUILD_IF(NELEMENTS(MANUAL_CONTROL_NAMES) != SYSTEMALARMS_MANUALCONTROL_NUMELEM, AlarmArrayMismatch);

static const char boot_reason_names[][15] = {
	[SYSTEMALARMS_REBOOTCAUSE_BROWNOUT] = "BOOT:BROWNOUT",
	[SYSTEMALARMS_REBOOTCAUSE_PINRESET] = "BOOT:PIN RESET",
	[SYSTEMALARMS_REBOOTCAUSE_POWERONRESET] = "BOOT:PWR ON RST",
	[SYSTEMALARMS_REBOOTCAUSE_SOFTWARERESET] = "BOOT:SW RESET",
	[SYSTEMALARMS_REBOOTCAUSE_INDEPENDENTWATCHDOG] "BOOT:INDY WDOG",
	[SYSTEMALARMS_REBOOTCAUSE_WINDOWWATCHDOG] = "BOOT:WIN WDOG",
	[SYSTEMALARMS_REBOOTCAUSE_LOWPOWER] = "BOOT:LOW POWER",
	[SYSTEMALARMS_REBOOTCAUSE_UNDEFINED] = "BOOT:UNDEFINED",
};

DONT_BUILD_IF(sizeof(*boot_reason_names) >= MAX_ALARM_LEN, BufferOverflow);

#define ALARM_OK 0
#define ALARM_WARN 1
#define ALARM_ERROR 2
#define ALARM_CRIT 3

static void msp_send_alarms(struct msp_bridge *m) {
	union {
		uint8_t buf[0];
		struct {
			uint8_t state;
			uint8_t msg[MAX_ALARM_LEN];
		} __attribute__((packed)) alarm;
	} data;

	SystemAlarmsData alarm;
	SystemAlarmsGet(&alarm);

	// Special case early boot times -- just report boot reason
	if (PIOS_Thread_Systime() < BOOT_DISPLAY_TIME_MS) {
		data.alarm.state = ALARM_CRIT;
		strncpy((char*)data.alarm.msg,
			(const char*)boot_reason_names[alarm.RebootCause],
			sizeof(*boot_reason_names));
		data.alarm.msg[sizeof(*boot_reason_names)] = 0;
		msp_send(m, MSP_ALARMS, data.buf, strlen((char*)data.alarm.msg)+1);
		return;
	}

	data.alarm.state = ALARM_OK;

	for (int i = 0; i < SYSTEMALARMS_ALARM_NUMELEM; i++) {
		if (alarm_names[i][0] != 0 &&
		    ((alarm.Alarm[i] == SYSTEMALARMS_ALARM_WARNING) ||
		     (alarm.Alarm[i] == SYSTEMALARMS_ALARM_ERROR) ||
		     (alarm.Alarm[i] == SYSTEMALARMS_ALARM_CRITICAL))) {

			uint8_t state = ALARM_OK;
			switch (alarm.Alarm[i]) {
			case SYSTEMALARMS_ALARM_WARNING:
				state = ALARM_WARN;
				break;
			case SYSTEMALARMS_ALARM_ERROR:
				state = ALARM_ERROR;
				break;
			case SYSTEMALARMS_ALARM_CRITICAL:
				state = ALARM_CRIT;
			}

			// Only take this alarm if it's worse than the previous one.
			if (state > data.alarm.state) {
				data.alarm.state = state;
				switch (i) {
				case SYSTEMALARMS_ALARM_SYSTEMCONFIGURATION:
					strncpy((char*)data.alarm.msg,
						(const char*)config_error_names[alarm.ConfigError],
						sizeof(*config_error_names));
					data.alarm.msg[sizeof(*config_error_names)] = 0;
					break;
				case SYSTEMALARMS_ALARM_MANUALCONTROL:
					strncpy((char*)data.alarm.msg,
						(const char*)manual_control_names[alarm.ManualControl],
						sizeof(*manual_control_names));
					data.alarm.msg[sizeof(*manual_control_names)] = 0;
					break;
				default:
					strncpy((char*)data.alarm.msg,
						(const char*)alarm_names[i], sizeof(*alarm_names));
					data.alarm.msg[sizeof(*alarm_names)] = 0;
				}
			}
		}
	}

	msp_send(m, MSP_ALARMS, data.buf, strlen((char*)data.alarm.msg)+1);
}

static msp_state msp_state_checksum(struct msp_bridge *m, uint8_t b)
{
	if ((m->checksum ^ b) != 0) {
		return MSP_IDLE;
	}

	// Respond to interesting things.
	switch (m->cmd_id) {
	case MSP_IDENT:
		msp_send_ident(m);
		break;
	case MSP_RAW_GPS:
		msp_send_raw_gps(m);
		break;
	case MSP_COMP_GPS:
		msp_send_comp_gps(m);
		break;
	case MSP_ALTITUDE:
		msp_send_altitude(m);
		break;
	case MSP_ATTITUDE:
		msp_send_attitude(m);
		break;
	case MSP_STATUS:
		msp_send_status(m);
		break;
	case MSP_ANALOG:
		msp_send_analog(m);
		break;
	case MSP_RC:
		msp_send_channels(m);
		break;
	case MSP_BOXIDS:
		msp_send_boxids(m);
		break;
	case MSP_ALARMS:
		msp_send_alarms(m);
		break;
	}
	return MSP_IDLE;
}

static msp_state msp_state_discard(struct msp_bridge *m, uint8_t b)
{
	return m->cmd_i++ == m->cmd_size ? MSP_IDLE : MSP_DISCARD;
}

/**
 * Process incoming bytes from an MSP query thing.
 * @param[in] b received byte
 * @return true if we should continue processing bytes
 */
static bool msp_receive_byte(struct msp_bridge *m, uint8_t b)
{
	switch (m->state) {
	case MSP_IDLE:
		switch (b) {
		case '<': // uavtalk matching with 0x3c 0x2x 0xxx 0x0x
			m->state = MSP_MAYBE_UAVTALK2;
			break;
		case '$':
			m->state = MSP_HEADER_START;
			break;
		default:
			m->state = MSP_IDLE;
		}
		break;
	case MSP_HEADER_START:
		m->state = b == 'M' ? MSP_HEADER_M : MSP_IDLE;
		break;
	case MSP_HEADER_M:
		m->state = b == '<' ? MSP_HEADER_SIZE : MSP_IDLE;
		break;
	case MSP_HEADER_SIZE:
		m->state = msp_state_size(m, b);
		break;
	case MSP_HEADER_CMD:
		m->state = msp_state_cmd(m, b);
		break;
	case MSP_FILLBUF:
		m->state = msp_state_fill_buf(m, b);
		break;
	case MSP_CHECKSUM:
		m->state = msp_state_checksum(m, b);
		break;
	case MSP_DISCARD:
		m->state = msp_state_discard(m, b);
		break;
	case MSP_MAYBE_UAVTALK2:
		// e.g. 3c 20 1d 00
		// second possible uavtalk byte
		m->state = (b&0xf0) == 0x20 ? MSP_MAYBE_UAVTALK3 : MSP_IDLE;
		break;
	case MSP_MAYBE_UAVTALK3:
		// third possible uavtalk byte can be anything
		m->state = MSP_MAYBE_UAVTALK4;
		break;
	case MSP_MAYBE_UAVTALK4:
		m->state = MSP_IDLE;
		// If this looks like the fourth possible uavtalk byte, we're done
		if ((b & 0xf0) == 0) {
			PIOS_COM_TELEM_RF = m->com;
			return false;
		}
		break;
	}

	return true;
}

/**
 * Module start routine automatically called after initialization routine
 * @return 0 when was successful
 */
static int32_t uavoMSPBridgeStart(void)
{
	if (!module_enabled)
		return -1;

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

		msp = PIOS_malloc(sizeof(*msp));
		if (msp != NULL) {
			memset(msp, 0x00, sizeof(*msp));

			msp->com = pios_com_msp_id;

			setMSPSpeed(msp);

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
