/**
 ******************************************************************************
 *
 * @file       frsky_packing.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @brief      Packs UAVObjects into FrSKY Smart Port frames
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

#include "frsky_packing.h"

#include "modulesettings.h"
#include "misc_math.h"
#include "physical_constants.h"
#include "attitudeactual.h"
#include "baroaltitude.h"
#include "positionactual.h"
#include "velocityactual.h"
#include "flightbatterystate.h"
#include "flightbatterysettings.h"
#include "gpstime.h"
#include "homelocation.h"
#include "accels.h"
#include "flightstatus.h"
#include "airspeedactual.h"
#include "nedaccel.h"
#include "velocityactual.h"
#include "attitudeactual.h"
/**
 * Encode baro altitude value
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_altitude(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
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
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_gps_course(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
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
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_vario(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
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
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_current(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
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
 * @param[in] arg argument specified in frsky_value_items[], index of battery cell pair
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_cells(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if ((frsky->batt_cell_count == 0) || (frsky->batt_cell_count - 1) < (arg * 2))
		return false;
	if (test_presence_only)
		return true;

	float voltage = 0;
	FlightBatteryStateVoltageGet(&voltage);

	uint32_t cell_voltage = (uint32_t)((voltage * 500.0f) / frsky->batt_cell_count);
	*value = ((cell_voltage & 0xfff) << 8) | ((arg * 2) & 0x0f) | ((frsky->batt_cell_count << 4) & 0xf0);
	if (((int16_t)frsky->batt_cell_count - 1) >= (arg * 2 + 1))
		*value |= ((cell_voltage & 0xfff) << 20);

	return true;
}

/**
 * Encode GPS status as T1 value
 * Right-most two digits means visible satellite count, left-most digit has following meaning:
 * 1 - no GPS connected
 * 2 - no fix
 * 3 - 2D fix
 * 4 - 3D fix
 * 5 - 3D fix and HomeLocation is SET - should be safe for navigation
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value successfully encoded or presence test passed
 */
bool frsky_encode_t1(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (GPSPositionHandle() == NULL)
		return false;
	
	if (test_presence_only)
		return true;
	
	uint8_t hl_set = HOMELOCATION_SET_FALSE;
	
	if (HomeLocationHandle())
		HomeLocationSetGet(&hl_set);

	int32_t t1 = 0;
	switch (frsky->gps_position.Status) {
	case GPSPOSITION_STATUS_NOGPS:
		t1 = 100;
		break;
	case GPSPOSITION_STATUS_NOFIX:
		t1 = 200;
		break;
	case GPSPOSITION_STATUS_FIX2D:
		t1 = 300;
		break;
	case GPSPOSITION_STATUS_FIX3D:
	case GPSPOSITION_STATUS_DIFF3D:
		if (hl_set == HOMELOCATION_SET_TRUE)
			t1 = 500;
		else
			t1 = 400;
		break;
	}
	if (frsky->gps_position.Satellites > 0)
		t1 += frsky->gps_position.Satellites;

	*value = (uint32_t)t1;

	return true;
}

/**
 * Encode GPS hDop and vDop as T2
 * Bits 0-7  = hDop * 100, max 255 (hDop = 2.55)
 * Bits 8-15 = vDop * 100, max 255 (vDop = 2.55)
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value successfully encoded or presence test passed
 */
bool frsky_encode_t2(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (GPSPositionHandle() == NULL)
		return false;

	if (test_presence_only)
		return true;

	uint32_t hdop = (uint32_t)(frsky->gps_position.HDOP * 100.0f);

	if (hdop > 255)
		hdop = 255;
			
	uint32_t vdop = (uint32_t)(frsky->gps_position.VDOP * 100.0f);
			
	if (vdop > 255)
		vdop = 255;
	
	*value = 256 * vdop + hdop;

	return true;
}

/**
 * Encode consumed battery energy as fuel
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_fuel(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
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
 * Encode configured values
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in frsky_value_items[]; 0=X, 1=Y, 2=Z
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_acc(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
{
	uint8_t accelDataSettings;
	ModuleSettingsFrskyAccelDataGet(&accelDataSettings);

	float acc = 0;
	switch(accelDataSettings) {
	case MODULESETTINGS_FRSKYACCELDATA_ACCELS: {
		if (AccelsHandle() == NULL)
			return false;
		else if (test_presence_only)
			return true;

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
		break;
	}

	case MODULESETTINGS_FRSKYACCELDATA_NEDACCELS: {
		if (NedAccelHandle() == NULL)
			return false;
		else if (test_presence_only)
			return true;

		switch (arg) {
		case 0:
			NedAccelNorthGet(&acc);
			break;
		case 1:
			NedAccelEastGet(&acc);
			break;
		case 2:
			NedAccelDownGet(&acc);
			break;
		}

		acc /= GRAVITY;
		acc *= 100.0f;
		break;
	}

	case MODULESETTINGS_FRSKYACCELDATA_NEDVELOCITY: {
		if (VelocityActualHandle() == NULL)
			return false;
		else if (test_presence_only)
			return true;

		switch (arg) {
		case 0:
			VelocityActualNorthGet(&acc);
			break;
		case 1:
			VelocityActualEastGet(&acc);
			break;
		case 2:
			VelocityActualDownGet(&acc);
			break;
		}

		acc *= 100.0f;
		break;
	}

	case MODULESETTINGS_FRSKYACCELDATA_ATTITUDEANGLES: {
		if (AttitudeActualHandle() == NULL)
			return false;
		else if (test_presence_only)
			return true;

		switch (arg) {
		case 0:
			AttitudeActualRollGet(&acc);
			break;
		case 1:
			AttitudeActualPitchGet(&acc);
			break;
		case 2:
			AttitudeActualYawGet(&acc);
			break;
		}

		acc *= 100.0f;
		break;
	}
	}

	int32_t frsky_acc = (int32_t) acc;
	*value = (uint32_t) frsky_acc;

	return true;
}

/**
 * Encode gps coordinates
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in frsky_value_items[]; 0=lattitude, 1=longitude
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_gps_coord(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
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
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_gps_alt(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
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
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_gps_speed(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
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
 * @param[in] arg argument specified in frsky_value_items[]; 0=date, 1=time
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_gps_time(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
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
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_rpm(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
{
	if (FlightStatusHandle() == NULL)
		return false;
	if (test_presence_only)
		return true;

	FlightStatusData flight_status;
	FlightStatusGet(&flight_status);

	*value = (flight_status.Armed == FLIGHTSTATUS_ARMED_ARMED) ? 200 : 100;
	*value += flight_status.FlightMode;

	return true;
}

/**
 * Encode true airspeed(TAS)
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value succesfully encoded or presence test passed
 */
bool frsky_encode_airspeed(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
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
 * Performs byte stuffing and checksum calculation
 * @param[out] obuff buffer where byte stuffed data will came in
 * @param[in,out] chk checksum byte to update
 * @param[in] byte
 * @returns count of bytes inserted to obuff (1 or 2)
 */
uint8_t frsky_insert_byte(uint8_t *obuff, uint16_t *chk, uint8_t byte)
{
	/* checksum calculation is based on data before byte-stuffing */
	*chk += byte;
	*chk += (*chk) >> 8;
	*chk &= 0x00ff;
	*chk += (*chk) >> 8;
	*chk &= 0x00ff;

	if (byte == 0x7e || byte == 0x7d) {
		obuff[0] = 0x7d;
		obuff[1] = byte &= ~0x20;
		return 2;
	}

	obuff[0] = byte;
	return 1;
}

/**
 * Send u32 value dataframe to FrSky SmartPort bus
 * @param[in] id FrSky value ID
 * @param[in] value value
 */
int32_t frsky_send_frame(uintptr_t com, enum frsky_value_id id, uint32_t value,
		bool send_prelude)
{
	/* each call of frsky_insert_byte can add 2 bytes to the buffer at maximum
	 * and therefore the worst-case is 17 bytes total (the first byte 0x10 won't be
	 * escaped) */
	uint8_t tx_data[17];
	uint16_t chk = 0;
	uint8_t cnt = 0;

	if (send_prelude) {
		tx_data[0] = 0x7e;
		tx_data[1] = 0x98;
		cnt = 2;
	}

	cnt += frsky_insert_byte(&tx_data[cnt], &chk, 0x10);
	cnt += frsky_insert_byte(&tx_data[cnt], &chk, (uint16_t)id & 0xff);
	cnt += frsky_insert_byte(&tx_data[cnt], &chk, ((uint16_t)id >> 8) & 0xff);
	cnt += frsky_insert_byte(&tx_data[cnt], &chk, value & 0xff);
	cnt += frsky_insert_byte(&tx_data[cnt], &chk, (value >> 8) & 0xff);
	cnt += frsky_insert_byte(&tx_data[cnt], &chk, (value >> 16) & 0xff);
	cnt += frsky_insert_byte(&tx_data[cnt], &chk, (value >> 24) & 0xff);
	cnt += frsky_insert_byte(&tx_data[cnt], &chk, 0xff - chk);

	PIOS_COM_SendBuffer(com, tx_data, cnt);

	return cnt;
}

/**
 * @}
 * @}
 */
