/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{
 * @addtogroup UAVOFrSKYSPortBridge UAVO to FrSKY S.PORT Bridge Module
 * @{
 *
 * @file       uavofrskysportbridge.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      Bridges selected UAVObjects to FrSKY Smart Port bus
 *
 * Since there is no public documentation of SmartPort protocol available,
 * this was put together by studying OpenTx source code, reading
 * tidbits of informations on the Internet and experimenting.
 * @see https://github.com/opentx/opentx/tree/next/radio/src/telemetry
 * @see https://code.google.com/p/telemetry-convert/wiki/FrSkySPortProtocol
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

#include "pios.h"
#include "openpilot.h"
#include "modulesettings.h"
#include "misc_math.h"
#include "physical_constants.h"
#include "attitudeactual.h"
#include "baroaltitude.h"
#include "positionactual.h"
#include "velocityactual.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "gpstime.h"
#include "homelocation.h"
#include "accels.h"
#include "flightstatus.h"
#include "airspeedactual.h"

#if defined(PIOS_INCLUDE_FRSKY_SPORT_TELEMETRY)

static void uavoFrSkySPortBridgeTask(void *parameters);
static bool frsky_encode_gps_course(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_altitude(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_vario(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_current(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_cells(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_t1(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_t2(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_fuel(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_acc(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_gps_coord(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_gps_alt(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_gps_speed(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_gps_time(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_rpm(uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_airspeed(uint32_t *value, bool test_presence_only, uint32_t arg);

#define FRSKY_POLL_REQUEST                 0x7e
#define FRSKY_MINIMUM_POLL_INTERVAL        10000

enum frsky_value_id {
	FRSKY_ALT_ID = 0x0100,
	FRSKY_VARIO_ID = 0x110,
	FRSKY_CURR_ID = 0x0200,
	FRSKY_VFAS_ID = 0x0210,
	FRSKY_CELLS_ID = 0x0300,
	FRSKY_T1_ID = 0x0400,
	FRSKY_T2_ID = 0x0410,
	FRSKY_RPM_ID = 0x0500,
	FRSKY_FUEL_ID = 0x0600,
	FRSKY_ACCX_ID = 0x0700,
	FRSKY_ACCY_ID = 0x0710,
	FRSKY_ACCZ_ID = 0x0720,
	FRSKY_GPS_LON_LAT_ID = 0x0800,
	FRSKY_GPS_ALT_ID = 0x0820,
	FRSKY_GPS_SPEED_ID = 0x0830,
	FRSKY_GPS_COURSE_ID = 0x0840,
	FRSKY_GPS_TIME_ID = 0x0850,
	FRSKY_ADC3_ID = 0x0900,
	FRSKY_ADC4_ID = 0x0910,
	FRSKY_AIR_SPEED_ID = 0x0a00,
	FRSKY_RSSI_ID = 0xf101,
	FRSKY_ADC1_ID = 0xf102,
	FRSKY_ADC2_ID = 0xf103,
	FRSKY_BATT_ID = 0xf104,
	FRSKY_SWR_ID = 0xf105,
};

enum frsky_state {
	FRSKY_STATE_WAIT_POLL_REQUEST,
	FRSKY_STATE_WAIT_SENSOR_ID,
	FRSKY_STATE_WAIT_TX_DONE,
};

struct frsky_value_item {
	enum frsky_value_id id;
	uint16_t period_ms;
	bool (*encode_value)(uint32_t *value, bool test_presence_only, uint32_t arg);
	uint32_t fn_arg;
};

static const struct frsky_value_item frsky_value_items[] = {
	{FRSKY_GPS_COURSE_ID,  100,   frsky_encode_gps_course, 0}, // attitude yaw estimate
	{FRSKY_ALT_ID,         100,   frsky_encode_altitude,   0}, // altitude estimate
	{FRSKY_VARIO_ID,       100,   frsky_encode_vario,      0}, // vertical speed
	{FRSKY_CURR_ID,        300,   frsky_encode_current,    0}, // battery current
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      0}, // battery cells 1-2
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      1}, // battery cells 3-4
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      2}, // battery cells 5-6
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      3}, // battery cells 7-8
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      4}, // battery cells 9-10
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      5}, // battery cells 11-12
	{FRSKY_T1_ID,          2000,  frsky_encode_t1,         0}, // baro temperature
	{FRSKY_T2_ID,          1500,  frsky_encode_t2,         0}, // encodes GPS status!
	{FRSKY_FUEL_ID,        600,   frsky_encode_fuel,       0}, // consumed battery energy
	{FRSKY_ACCX_ID,        250,   frsky_encode_acc,        0}, // accX
	{FRSKY_ACCY_ID,        250,   frsky_encode_acc,        1}, // accY
	{FRSKY_ACCZ_ID,        250,   frsky_encode_acc,        2}, // accZ
	{FRSKY_GPS_LON_LAT_ID, 100,   frsky_encode_gps_coord,  0}, // lattitude
	{FRSKY_GPS_LON_LAT_ID, 100,   frsky_encode_gps_coord,  1}, // longitude
	{FRSKY_GPS_ALT_ID,     750,   frsky_encode_gps_alt,    0}, // gps altitude
	{FRSKY_GPS_SPEED_ID,   200,   frsky_encode_gps_speed,  0}, // gps speed
	{FRSKY_GPS_TIME_ID,    8000,  frsky_encode_gps_time,   0}, // gps date
	{FRSKY_GPS_TIME_ID,    2000,  frsky_encode_gps_time,   1}, // gps time
	{FRSKY_RPM_ID,         1500,  frsky_encode_rpm,        0}, // encodes flight status!
	{FRSKY_AIR_SPEED_ID,   100,   frsky_encode_airspeed,   0}, // airspeed
};

static const uint8_t frsky_sensor_ids[] = {0x1b, 0x0d, /*0x34, 0x67*/};
struct frsky_sport_telemetry {
	xTaskHandle task;
	uintptr_t com;
	uint32_t item_last_triggered[NELEMENTS(frsky_value_items)];
	uint32_t last_poll_time;
	uint8_t ignore_rx_chars;
	enum frsky_state state;
	int32_t scheduled_item;
	bool use_current_sensor;
	uint8_t batt_cell_count;
	bool use_baro_sensor;
	FlightBatterySettingsData battery_settings;
	GPSPositionData gps_position;
};

#define FRSKY_SPORT_BAUDRATE                    57600

#if defined(PIOS_FRSKY_SPORT_TELEMETRY_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_FRSKY_SPORT_TELEMETRY_STACK_SIZE
#else
#define STACK_SIZE_BYTES 672
#endif
#define TASK_PRIORITY               (tskIDLE_PRIORITY + 1)

static bool module_enabled;
static struct frsky_sport_telemetry *frsky;

/**
 * Encode baro altitude value
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_altitude(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (!frsky->use_baro_sensor || (PositionActualHandle() == NULL))
		return false;

	if (test_presence_only)
		return true;
	// instead of encoding baro altitude directly, we will use
	// more accurate estimation in PositionActual UAVO
	float down = 0;
	PositionActualDownGet(&down);
	int32_t alt = (int32_t)(-down * 100.0f);
	*value = (uint32_t) alt;
	return true;
}

/**
 * Encode heading value
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_gps_course(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (AttitudeActualHandle() == NULL)
		return false;

	if (test_presence_only)
		return true;

	AttitudeActualData attitude;
	AttitudeActualGet(&attitude);
	float hdg = (attitude.Yaw >= 0) ? attitude.Yaw : (attitude.Yaw + 360.0f);
	*value = (uint32_t)(hdg * 100.0f);

	return true;
}

/**
 * Encode vertical speed value
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_vario(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (!frsky->use_baro_sensor || VelocityActualHandle() == NULL)
		return false;

	if (test_presence_only)
		return true;

	float down = 0;
	VelocityActualDownGet(&down);
	int32_t vspeed = (int32_t)(-down * 100.0f);
	*value = (uint32_t) vspeed;

	return true;
}

/**
 * Encode battery current value
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_current(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (!frsky->use_current_sensor)
		return false;
	if (test_presence_only)
		return true;

	float current = 0;
	FlightBatteryStateCurrentGet(&current);
	int32_t current_frsky = (int32_t)(current * 10.0f);
	*value = (uint32_t) current_frsky;

	return true;
}

/**
 * Encode battery cells voltage
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[], index of battery cell pair
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_cells(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if ((frsky->batt_cell_count == 0) || (frsky->batt_cell_count - 1) < (arg * 2))
		return false;
	if (test_presence_only)
		return true;

	float voltage = 0;
	FlightBatteryStateVoltageGet(&voltage);

	uint32_t cell_voltage = (uint32_t)((voltage * 500.0f) / frsky->batt_cell_count);
	*value = ((cell_voltage & 0xfff) << 8) | ((arg * 2) & 0x0f);
	if (((int16_t)frsky->batt_cell_count - 1) >= (arg * 2 + 1))
		*value |= ((cell_voltage & 0xfff) << 20);

	return true;
}

/**
 * Encode temperature of barosensor as T1
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_t1(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (!frsky->use_baro_sensor)
		return false;
	if (test_presence_only)
		return true;

	float temp = 0;
	BaroAltitudeTemperatureGet(&temp);
	int32_t t1 = (int32_t)temp;
	*value = (uint32_t)t1;

	return true;
}

/**
 * Encode GPS status as T2 value
 * Right-most two digits means visible satellite count, left-most digit has following meaning:
 * 0 - no GPS connected
 * 1 - no fix
 * 2 - 2D fix
 * 3 - 3D fix
 * 4 - 3D fix and HomeLocation is SET - should be safe for navigation
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_t2(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (GPSPositionHandle() == NULL)
		return false;
	if (test_presence_only)
		return true;
	uint8_t hl_set = HOMELOCATION_SET_FALSE;
	if (HomeLocationHandle())
		HomeLocationSetGet(&hl_set);

	int32_t t2 = 0;
	switch (frsky->gps_position.Status) {
	case GPSPOSITION_STATUS_NOGPS:
		t2 = 0;
		break;
	case GPSPOSITION_STATUS_NOFIX:
		t2 = 100;
		break;
	case GPSPOSITION_STATUS_FIX2D:
		t2 = 200;
		break;
	case GPSPOSITION_STATUS_FIX3D:
	case GPSPOSITION_STATUS_DIFF3D:
		if (hl_set == HOMELOCATION_SET_TRUE)
			t2 = 400;
		else
			t2 = 300;
		break;
	}
	if (frsky->gps_position.Satellites > 0)
		t2 += frsky->gps_position.Satellites;

	*value = (uint32_t)t2;

	return true;
}

/**
 * Encode consumed battery energy as fuel
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_fuel(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (!frsky->use_current_sensor)
		return false;
	if (test_presence_only)
		return true;

	uint32_t capacity = frsky->battery_settings.Capacity;
	float consumed_mahs = 0;
	FlightBatteryStateConsumedEnergyGet(&consumed_mahs);

	float fuel =  (uint32_t)(100.0f * (1.0f - consumed_mahs / capacity));
	fuel = bound_min_max(fuel, 0.0f, 100.0f);
	*value = (uint32_t) fuel;

	return true;
}

/**
 * Encode accelerometer values
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]; 0=X, 1=Y, 2=Z
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_acc(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (AccelsHandle() == NULL)
		return false;
	if (test_presence_only)
		return true;

	float acc = 0;
	switch (arg) {
	case 0:
		AccelsxGet(&acc);
		break;
	case 1:
		AccelsyGet(&acc);
		break;
	case 2:
		AccelszGet(&acc);
		break;
	}

	acc /= GRAVITY;
	acc *= 100.0f;

	int32_t frsky_acc = (int32_t) acc;
	*value = (uint32_t) frsky_acc;

	return true;
}

/**
 * Encode gps coordinates
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]; 0=lattitude, 1=longitude
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_gps_coord(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (GPSPositionHandle() == NULL)
		return false;
	if (frsky->gps_position.Status == GPSPOSITION_STATUS_NOFIX
			|| frsky->gps_position.Status == GPSPOSITION_STATUS_NOGPS)
		return false;
	if (test_presence_only)
		return true;

	uint32_t frsky_coord = 0;
	int32_t coord = 0;
	if (arg == 0) {
		// lattitude
		coord = frsky->gps_position.Latitude;
		if (coord >= 0)
			frsky_coord = 0;
		else
			frsky_coord = 1 << 30;
	} else {
		// longitude
		coord = frsky->gps_position.Longitude;
		if (coord >= 0)
			frsky_coord = 2 << 30;
		else
			frsky_coord = 3 << 30;
	}
	coord = abs(coord);
	frsky_coord |= (((uint64_t)coord * 6ull) / 100);

	*value = frsky_coord;
	return true;
}

/**
 * Encode gps altitude
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_gps_alt(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (GPSPositionHandle() == NULL)
		return false;
	if (frsky->gps_position.Status != GPSPOSITION_STATUS_FIX3D
			&& frsky->gps_position.Status != GPSPOSITION_STATUS_DIFF3D)
		return false;
	if (test_presence_only)
		return true;

	int32_t frsky_gps_alt = (int32_t)(frsky->gps_position.Altitude * 100.0f);
	*value = (uint32_t)frsky_gps_alt;

	return true;
}

/**
 * Encode gps speed
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_gps_speed(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (GPSPositionHandle() == NULL)
		return false;
	if (frsky->gps_position.Status != GPSPOSITION_STATUS_FIX3D
			&& frsky->gps_position.Status != GPSPOSITION_STATUS_DIFF3D)
		return false;
	if (test_presence_only)
		return true;

	int32_t frsky_speed = (int32_t)((frsky->gps_position.Groundspeed / KNOTS2M_PER_SECOND) * 1000);
	*value = frsky_speed;
	return true;
}

/**
 * Encode GPS UTC time
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]; 0=date, 1=time
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_gps_time(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (GPSPositionHandle() == NULL || GPSTimeHandle() == NULL)
		return false;
	if (frsky->gps_position.Status != GPSPOSITION_STATUS_FIX3D
			&& frsky->gps_position.Status != GPSPOSITION_STATUS_DIFF3D)
		return false;
	if (test_presence_only)
		return true;

	GPSTimeData gps_time;
	GPSTimeGet(&gps_time);
	uint32_t frsky_time = 0;

	if (arg == 0) {
		// encode date
		frsky_time = 0x000000ff;
		frsky_time |= gps_time.Day << 8;
		frsky_time |= gps_time.Month << 16;
		frsky_time |= (gps_time.Year % 100) << 24;
	} else {
		frsky_time = 0;
		frsky_time |= gps_time.Second << 8;
		frsky_time |= gps_time.Minute << 16;
		frsky_time |= gps_time.Hour << 24;
	}
	*value = frsky_time;
	return true;
}

/**
 * Encodes ARM status and flight mode number as RPM value
 * Since there is no RPM information in any UAVO available,
 * we will intentionaly misuse this item to encode another useful information.
 * It will encode flight status as three-digit number as follow:
 * most left digit encodes arm status (1=armed, 0=disarmed)
 * two most right digits encodes flight mode number (see FlightStatus UAVO FlightMode enum)
 * To work this propperly on Taranis, you have to set Blades to "1" in telemetry setting
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_rpm(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (FlightStatusHandle() == NULL)
		return false;
	if (test_presence_only)
		return true;

	FlightStatusData flight_status;
	FlightStatusGet(&flight_status);

	*value = (flight_status.Armed == FLIGHTSTATUS_ARMED_ARMED) ? 100 : 0;
	*value += flight_status.FlightMode;

	return true;
}

/**
 * Encode true airspeed(TAS)
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in value_records[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_airspeed(uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (AirspeedActualHandle() == NULL)
		return false;
	if (test_presence_only)
		return true;

	AirspeedActualData airspeed;
	AirspeedActualGet(&airspeed);
	int32_t frsky_speed = (int32_t)((airspeed.TrueAirspeed / KNOTS2M_PER_SECOND) * 10);
	*value = (uint32_t)frsky_speed;

	return true;
}

/**
 * Send u32 value dataframe to FrSky SmartPort bus
 * @param[in] id FrSky value ID
 * @param[in] value value
 */
static void frsky_send_frame(enum frsky_value_id id, uint32_t value)
{
	uint8_t tx_data[8];
	uint16_t chk = 0;
	tx_data[0] = 0x10;		// FrSky data frame
	tx_data[1] = (uint16_t)id & 0xff;
	tx_data[2] = ((uint16_t)id >> 8) & 0xff;
	tx_data[3] = value & 0xff;
	tx_data[4] = (value >> 8) & 0xff;
	tx_data[5] = (value >> 16) & 0xff;
	tx_data[6] = (value >> 24) & 0xff;

	uint8_t i;
	for (i = 0; i < 7; i++) {
		chk += tx_data[i];
		chk += chk >> 8;
		chk &= 0x00ff;
		chk += chk >> 8;
		chk &= 0x00ff;
	}
	tx_data[7] = 0xff - chk;

	PIOS_COM_SendBuffer(frsky->com, tx_data, 8);

	frsky->ignore_rx_chars = 8;
}

/**
 * Scan for value item with the longest expired time and schedule it to send in next poll turn
 *
 */
static void frsky_schedule_next_item(void)
{
	uint32_t i = 0;
	int32_t max_exp_time = INT32_MIN;
	int32_t most_exp_item = -1;
	for (i = 0; i < NELEMENTS(frsky_value_items); i++) {
		if (frsky_value_items[i].encode_value(0, true, frsky_value_items[i].fn_arg)) {
			int32_t exp_time = PIOS_DELAY_GetuSSince(frsky->item_last_triggered[i]) -
					(frsky_value_items[i].period_ms * 1000);
			if (exp_time > max_exp_time) {
				max_exp_time = exp_time;
				most_exp_item = i;
			}
		}
	}
	frsky->scheduled_item = most_exp_item;
}
/**
 * Send value item previously scheduled by frsky_schedule_next_itme()
 * @returns true when item value was sended
 */
static bool frsky_send_scheduled_item(void)
{
	int32_t item = frsky->scheduled_item;
	if ((item >= 0) && (item < NELEMENTS(frsky_value_items))) {
		frsky->item_last_triggered[item] = PIOS_DELAY_GetuS();
		uint32_t value = 0;
		if (frsky_value_items[item].encode_value(&value, false,
				frsky_value_items[item].fn_arg)) {
			frsky_send_frame((uint16_t)(frsky_value_items[item].id), value);
			return true;
		}
	}

	return false;
}

/**
 * Process incoming bytes from FrSky S.PORT bus
 * @param[in] b received byte
 */
static void frsky_receive_byte(uint8_t b)
{
	uint32_t i = 0;
	switch (frsky->state) {
	case FRSKY_STATE_WAIT_TX_DONE:
		// because RX and TX are connected, we need to ignore bytes
		// transmited by us
		if (frsky->ignore_rx_chars)
			frsky->ignore_rx_chars--;
		else
			frsky->state = FRSKY_STATE_WAIT_POLL_REQUEST;
		break;

	case FRSKY_STATE_WAIT_POLL_REQUEST:
		if (b == FRSKY_POLL_REQUEST) {
			// X8R is polling us every 13ms
			if (PIOS_DELAY_GetuSSince(frsky->last_poll_time) > FRSKY_MINIMUM_POLL_INTERVAL) {
				frsky->last_poll_time = PIOS_DELAY_GetuS();
				frsky->state = FRSKY_STATE_WAIT_SENSOR_ID;
			}
		}
		break;

	case FRSKY_STATE_WAIT_SENSOR_ID:
		frsky->state = FRSKY_STATE_WAIT_POLL_REQUEST;
		for (i = 0; i < sizeof(frsky_sensor_ids); i++) {
			if (frsky_sensor_ids[i] == b) {
				// get GPSPositionData once here to be more efficient, since
				// GPS position data are very often used by encode() handlers
				if (GPSPositionHandle() != NULL)
					GPSPositionGet(&frsky->gps_position);
				// send item previously scheduled
				if (frsky_send_scheduled_item())
					frsky->state = FRSKY_STATE_WAIT_TX_DONE;
				// schedule item for next poll turn
				frsky_schedule_next_item();
				break;
			}
		}
		break;
	}
}

/**
 * Module start routine automatically called after initialization routine
 * @return 0 when was successful
 */
static int32_t uavoFrSkySPortBridgeStart(void)
{
	if (!module_enabled)
		return -1;

	if (FlightBatterySettingsHandle() != NULL
			&& FlightBatteryStateHandle() != NULL) {
		uint8_t sensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_NUMELEM];
		FlightBatterySettingsSensorTypeGet(sensorType);
		if (sensorType[FLIGHTBATTERYSETTINGS_SENSORTYPE_BATTERYCURRENT]
				== FLIGHTBATTERYSETTINGS_SENSORTYPE_ENABLED)
			frsky->use_current_sensor = true;
		FlightBatterySettingsGet(&frsky->battery_settings);
		frsky->batt_cell_count = frsky->battery_settings.NbCells;
	}
	if (BaroAltitudeHandle() != NULL
			&& PIOS_SENSORS_GetQueue(PIOS_SENSOR_BARO) != NULL)
		frsky->use_baro_sensor = true;

	xTaskCreate(uavoFrSkySPortBridgeTask, (signed char *)"uavoFrSkySPortBridge",
			STACK_SIZE_BYTES / 4, NULL, TASK_PRIORITY,
			&frsky->task);
	TaskMonitorAdd(TASKINFO_RUNNING_UAVOFRSKYSPORTBRIDGE,
			frsky->task);

	return 0;
}

/**
 * Module initialization routine
 * @return 0 when initialization was successful
 */
static int32_t uavoFrSkySPortBridgeInitialize(void)
{
	uintptr_t sport_com = PIOS_COM_FRSKY_SPORT;

	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);

	if (sport_com && (module_state[MODULESETTINGS_ADMINSTATE_UAVOFRSKYSPORTBRIDGE]
						== MODULESETTINGS_ADMINSTATE_ENABLED)) {
		frsky = PIOS_malloc(sizeof(struct frsky_sport_telemetry));
		if (frsky != NULL) {
			memset(frsky, 0x00, sizeof(struct frsky_sport_telemetry));

			frsky->com = sport_com;
			frsky->ignore_rx_chars = 0;
			frsky->last_poll_time = PIOS_DELAY_GetuS();
			frsky->state = FRSKY_STATE_WAIT_POLL_REQUEST;
			frsky->scheduled_item = -1;
			frsky->use_current_sensor = false;
			frsky->batt_cell_count = 0;
			frsky->use_baro_sensor = false;

			uint8_t i;
			for (i = 0; i < NELEMENTS(frsky_value_items); i++)
				frsky->item_last_triggered[i] = PIOS_DELAY_GetuS();
			PIOS_COM_ChangeBaud(frsky->com, FRSKY_SPORT_BAUDRATE);
			module_enabled = true;
			return 0;
		}
	}

	module_enabled = false;

	return -1;
}
MODULE_INITCALL(uavoFrSkySPortBridgeInitialize, uavoFrSkySPortBridgeStart)

/**
 * Main task routine
 * @param[in] parameters parameter given by xTaskCreate()
 */
static void uavoFrSkySPortBridgeTask(void *parameters)
{
	while (1) {
		uint8_t b = 0;
		uint16_t count = PIOS_COM_ReceiveBuffer(frsky->com, &b, 1, portMAX_DELAY);
		if (count)
			frsky_receive_byte(b);
	}
}

#endif //PIOS_INCLUDE_FRSKY_SPORT_TELEMETRY
/**
 * @}
 * @}
 */
