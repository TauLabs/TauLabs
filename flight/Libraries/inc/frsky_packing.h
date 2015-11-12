/**
 ******************************************************************************
 *
 * @file       frsky_packing.h
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

#ifndef FRSKY_PACKING_H
#define FRSKY_PACKING_H

#include "pios.h"
#include "openpilot.h"

#include "flightbatterysettings.h"
#include "gpsposition.h"

struct frsky_settings {
	bool use_current_sensor;
	uint8_t batt_cell_count;
	bool use_baro_sensor;
	FlightBatterySettingsData battery_settings;
	GPSPositionData gps_position;
};

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

struct frsky_value_item {
	enum frsky_value_id id;
	uint16_t period_ms;
	bool (*encode_value)(struct frsky_settings *sport, uint32_t *value, bool test_presence_only, uint32_t arg);
	uint32_t fn_arg;
};

bool frsky_encode_gps_course(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_altitude(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_vario(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_current(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_cells(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_t1(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_t2(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_fuel(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_acc(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_gps_coord(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_gps_alt(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_gps_speed(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_gps_time(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_rpm(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
bool frsky_encode_airspeed(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
uint8_t frsky_insert_byte(uint8_t *obuff, uint16_t *chk, uint8_t byte);
int32_t frsky_send_frame(uintptr_t com, enum frsky_value_id id, uint32_t value,
		bool send_prelude);

#endif /* FRSKY_PACKING_H */

/**
 * @}
 * @}
 */
