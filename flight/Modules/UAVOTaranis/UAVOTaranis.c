/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup UAVOTaranis UAVO to Taranis S.PORT converter
 * @{ 
 *
 * @file       uavoFrSKYSensorHubBridge.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @brief      Bridges selected UAVObjects to Taranis S.PORT
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

/*
 * This module is derived from UAVOFrSKYSportPort but differs
 * in a number of ways. Specifically the SPort code is expected
 * to listen for sensor requests and respond appropriately. This
 * code simply can spew data to the Taranis (since it is talking
 * to different harware). The scheduling system is reused between
 * the two, but duplicated because it isn't really part of the 
 * packing, and touches on the private state of the state machine
 * code in the module 
 */

#include "frsky_packing.h"
#include "pios_thread.h"

#include "openpilot.h"
#include "physical_constants.h"
#include "modulesettings.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "airspeedactual.h"
#include "baroaltitude.h"
#include "accels.h"
#include "positionactual.h"
#include "velocityactual.h"
#include "flightstatus.h"
#include "rfm22bstatus.h"

#if defined(PIOS_INCLUDE_TARANIS_SPORT)

static void uavoTaranisTask(void *parameters);

static bool frsky_encode_rssi(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_swr(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);
static bool frsky_encode_battery(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg);

#define FRSKY_POLL_REQUEST                 0x7e
#define FRSKY_MINIMUM_POLL_INTERVAL        10000

#define VOLT_RATIO (20)

enum frsky_state {
	FRSKY_STATE_WAIT_POLL_REQUEST,
	FRSKY_STATE_WAIT_SENSOR_ID,
	FRSKY_STATE_WAIT_TX_DONE,
};

static const struct frsky_value_item frsky_value_items[] = {
	{FRSKY_CURR_ID,        300,   frsky_encode_current,    0}, // battery current
	{FRSKY_BATT_ID,        200,   frsky_encode_battery,    0}, // send battery voltage
	{FRSKY_FUEL_ID,        200,   frsky_encode_fuel,       0}, // consumed battery energy
	{FRSKY_RSSI_ID,        100,   frsky_encode_rssi,       0}, // send RSSI information
	{FRSKY_SWR_ID,         100,   frsky_encode_swr,        0}, // send RSSI information
	{FRSKY_ALT_ID,         100,   frsky_encode_altitude,   0}, // altitude estimate
	{FRSKY_VARIO_ID,       100,   frsky_encode_vario,      0}, // vertical speed
	{FRSKY_RPM_ID,         1500,  frsky_encode_rpm,        0}, // encodes flight status!
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      0}, // battery cells 1-2
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      1}, // battery cells 3-4
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      2}, // battery cells 5-6
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      3}, // battery cells 7-8
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      4}, // battery cells 9-10
	{FRSKY_CELLS_ID,       850,   frsky_encode_cells,      5}, // battery cells 11-12
};

struct frsky_sport_telemetry {
	enum frsky_state state;
	int32_t scheduled_item;
	uint32_t last_poll_time;
	uint8_t ignore_rx_chars;
	uintptr_t com;
	struct frsky_settings frsky_settings;
	uint32_t item_last_triggered[NELEMENTS(frsky_value_items)];
};

#define FRSKY_SPORT_BAUDRATE                    57600

#if defined(PIOS_FRSKY_SPORT_TELEMETRY_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_FRSKY_SPORT_TELEMETRY_STACK_SIZE
#else
#define STACK_SIZE_BYTES 672
#endif
#define TASK_PRIORITY               PIOS_THREAD_PRIO_LOW

static struct pios_thread *uavoTaranisTaskHandle;
static bool module_enabled;
static struct frsky_sport_telemetry *frsky;

/**
 * Encode RSSI value
 * @param[out] value encoded value
 * @param[in] test_presence_only true when function should only test for availability of this value
 * @param[in] arg argument specified in frsky_value_items[]
 * @returns true when value succesfully encoded or presence test passed
 */
static bool frsky_encode_rssi(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
{
	uint8_t local_link_quality, local_link_connected;

	RFM22BStatusLinkStateGet(&local_link_connected);
	RFM22BStatusLinkQualityGet(&local_link_quality);

	RFM22BStatusData rfm22bStatus;
	RFM22BStatusInstGet(1, &rfm22bStatus);

	if (local_link_connected == RFM22BSTATUS_LINKSTATE_CONNECTED) {

		if (rfm22bStatus.LinkQuality == 127) {
			// When we are receiving all the packets, then move the RSSI (originally
			// in dBm ranging from -100 to -20) from 50% to 100%
			*value = (rfm22bStatus.RSSI + 100) * 50 / 80 + 50;
		} else {
			// If we are dropping packets then report the link quality, which will by
			// be at maximum 50%
			*value = (rfm22bStatus.LinkQuality * 100 / 256);
		}


	} else {
		*value = 0;
	}

	return true;
}

static bool frsky_encode_swr(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
{
	*value = 1;
	return true;
}

static bool frsky_encode_battery(struct frsky_settings *frsky, uint32_t *value, bool test_presence_only, uint32_t arg)
{
	float voltage = 0;
	FlightBatteryStateVoltageGet(&voltage);
	*value = (uint8_t) (voltage * VOLT_RATIO);

	return true;
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
		if (frsky_value_items[i].encode_value(&frsky->frsky_settings, 0, true, frsky_value_items[i].fn_arg)) {
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
		if (frsky_value_items[item].encode_value(&frsky->frsky_settings, &value, false,
				frsky_value_items[item].fn_arg)) {
			frsky_send_frame(frsky->com, (uint16_t)(frsky_value_items[item].id), value, true);
			return true;
		}
	}

	return false;
}

/**
 * Start the module
 * \return -1 if start failed
 * \return 0 on success
 */
static int32_t uavoTaranisStart(void)
{
	if (module_enabled) {
		// Start tasks
		uavoTaranisTaskHandle = PIOS_Thread_Create(
				uavoTaranisTask, "uavoFrSKYSensorHubBridge",
				STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
		TaskMonitorAdd(TASKINFO_RUNNING_UAVOFRSKYSBRIDGE,
				uavoTaranisTaskHandle);
		return 0;
	}
	return -1;
}

/**
 * Initialize the module
 * \return -1 if initialization failed
 * \return 0 on success
 */
static int32_t uavoTaranisInitialize(void)
{
	uint32_t sport_com = PIOS_COM_FRSKY_SPORT;

	if (sport_com) {


		frsky = PIOS_malloc(sizeof(struct frsky_sport_telemetry));
		if (frsky != NULL) {
			memset(frsky, 0x00, sizeof(struct frsky_sport_telemetry));

			// These objects are registered on the TLM so it
			// can intercept them from the telemetry stream
			FlightBatteryStateInitialize();
			FlightStatusInitialize();
			PositionActualInitialize();
			VelocityActualInitialize();

			frsky->frsky_settings.use_current_sensor = false;
			frsky->frsky_settings.batt_cell_count = 0;
			frsky->frsky_settings.use_baro_sensor = false;
			frsky->state = FRSKY_STATE_WAIT_POLL_REQUEST;
			frsky->last_poll_time = PIOS_DELAY_GetuS();
			frsky->ignore_rx_chars = 0;
			frsky->scheduled_item = -1;
			frsky->com = sport_com;

			uint8_t i;
			for (i = 0; i < NELEMENTS(frsky_value_items); i++)
				frsky->item_last_triggered[i] = PIOS_DELAY_GetuS();
			PIOS_COM_ChangeBaud(frsky->com, FRSKY_SPORT_BAUDRATE);
			module_enabled = true;
			return 0;
		}

		module_enabled = true;

		return 0;
	}

	module_enabled = false;

	return -1;
}
MODULE_INITCALL(uavoTaranisInitialize, uavoTaranisStart)

/**
 * Main task. It does not return.
 */
static void uavoTaranisTask(void *parameters)
{
	while(1) {

		if (true) {

			// for some reason, only first four messages are sent.
			for (uint32_t i = 0; i < sizeof(frsky_value_items) / sizeof(frsky_value_items[0]); i++) {
				frsky->scheduled_item = i;
				frsky_send_scheduled_item();
				PIOS_Thread_Sleep(5);
			}

		}

		if (false) { 

			// fancier schedlued message sending. doesn't appear to work
			// currently.
			PIOS_Thread_Sleep(1);
			frsky_schedule_next_item();
			frsky_send_scheduled_item();
		}

	}
}

#endif /* PIOS_INCLUDE_TARANIS_SPORT */
/**
 * @}
 * @}
 */
