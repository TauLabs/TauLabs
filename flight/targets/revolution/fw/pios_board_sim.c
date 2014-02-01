/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup Revolution OpenPilot revolution support files
 * @{
 *
 * @file       pios_board_sim.c 
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Simulation of the board specific initialization routines
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

#include <pios.h>
#include "pios_board_sim.h"
#include <pios_com_priv.h>
#include <pios_tcp_priv.h>
#include <pios_udp_priv.h>
#include <openpilot.h>
#include <uavobjectsinit.h>

#include "accels.h"
#include "baroaltitude.h"
#include "gpsposition.h"
#include "gyros.h"
#include "gyrosbias.h"
#include "magnetometer.h"
#include "manualcontrolsettings.h"

#include "pios_rcvr_priv.h"
#include "pios_gcsrcvr_priv.h"

void Stack_Change() {
}

const struct pios_tcp_cfg pios_tcp_telem_cfg = {
  .ip = "0.0.0.0",
  .port = 9000,
};

const struct pios_udp_cfg pios_udp_telem_cfg = {
	.ip = "0.0.0.0",
	.port = 9000,
};

const struct pios_tcp_cfg pios_tcp_gps_cfg = {
  .ip = "0.0.0.0",
  .port = 9001,
};
const struct pios_tcp_cfg pios_tcp_debug_cfg = {
  .ip = "0.0.0.0",
  .port = 9002,
};

#ifdef PIOS_COM_AUX
/*
 * AUX USART
 */
const struct pios_tcp_cfg pios_tcp_aux_cfg = {
  .ip = "0.0.0.0",
  .port = 9003,
};
#endif

#define PIOS_COM_TELEM_RF_RX_BUF_LEN 384
#define PIOS_COM_TELEM_RF_TX_BUF_LEN 384
#define PIOS_COM_GPS_RX_BUF_LEN 96

/**
 * Simulation of the flash filesystem
 */
#include "../../../tests/logfs/unittest_init.c"

uintptr_t pios_uavo_settings_fs_id;
uintptr_t pios_waypoints_settings_fs_id;

/*
 * Board specific number of devices.
 */
extern const struct pios_com_driver pios_serial_com_driver;
extern const struct pios_com_driver pios_udp_com_driver;
extern const struct pios_com_driver pios_tcp_com_driver;

uintptr_t pios_com_debug_id;
uintptr_t pios_com_telem_rf_id;
uintptr_t pios_com_telem_usb_id;
uintptr_t pios_com_gps_id;
uintptr_t pios_com_aux_id;
uintptr_t pios_com_spectrum_id;
uintptr_t pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE];

/**
 * PIOS_Board_Init()
 * initializes all the core systems on this specific hardware
 * called from System/openpilot.c
 */
void PIOS_Board_Init(void) {

	/* Delay system */
	PIOS_DELAY_Init();

	int32_t retval = PIOS_Flash_Posix_Init(&pios_posix_flash_id, &flash_config);
	if (retval != 0) {

	    /* create an empty, appropriately sized flash filesystem */
	    FILE * theflash = fopen("theflash.bin", "w");
	    uint8_t sector[flash_config.size_of_sector];
	    memset(sector, 0xFF, sizeof(sector));
	    for (uint32_t i = 0; i < flash_config.size_of_flash / flash_config.size_of_sector; i++) {
	      fwrite(sector, sizeof(sector), 1, theflash);
	    }
	    fclose(theflash);

		retval = PIOS_Flash_Posix_Init(&pios_posix_flash_id, &flash_config);

		if (retval != 0) {
			fprintf(stderr, "Unable to initialize flash posix simulator: %d\n", retval);
			exit(0);
		}
		
	}

	/* Register the partition table */
	PIOS_FLASH_register_partition_table(pios_flash_partition_table, NELEMENTS(pios_flash_partition_table));

	if (PIOS_FLASHFS_Logfs_Init(&pios_uavo_settings_fs_id, &flashfs_config_settings, FLASH_PARTITION_LABEL_SETTINGS) != 0)
		fprintf(stderr, "Unable to open the settings partition\n");

	if (PIOS_FLASHFS_Logfs_Init(&pios_waypoints_settings_fs_id, &flashfs_config_waypoints, FLASH_PARTITION_LABEL_WAYPOINTS) != 0)
		fprintf(stderr, "Unable to open the waypoints partition\n");

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();
	UAVObjectsInitializeAll();

	AccelsInitialize();
	BaroAltitudeInitialize();
	MagnetometerInitialize();
	GPSPositionInitialize();
	GyrosInitialize();
	GyrosBiasInitialize();

	/* Initialize the alarms library */
	AlarmsInitialize();

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

#if defined(PIOS_INCLUDE_COM)
#if defined(PIOS_INCLUDE_TELEMETRY_RF) && 1
	{
		uintptr_t pios_tcp_telem_rf_id;
		if (PIOS_TCP_Init(&pios_tcp_telem_rf_id, &pios_tcp_telem_cfg)) {
			PIOS_Assert(0);
		}

		uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_TELEM_RF_RX_BUF_LEN);
		uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_TELEM_RF_TX_BUF_LEN);
		PIOS_Assert(rx_buffer);
		PIOS_Assert(tx_buffer);
		if (PIOS_COM_Init(&pios_com_telem_rf_id, &pios_tcp_com_driver, pios_tcp_telem_rf_id,
						  rx_buffer, PIOS_COM_TELEM_RF_RX_BUF_LEN,
						  tx_buffer, PIOS_COM_TELEM_RF_TX_BUF_LEN)) {
			PIOS_Assert(0);
		}
	}
#endif /* PIOS_INCLUDE_TELEMETRY_RF */

#if defined(PIOS_INCLUDE_TELEMETRY_RF) && 0
	{
		uintptr_t pios_udp_telem_rf_id;
		if (PIOS_UDP_Init(&pios_udp_telem_rf_id, &pios_udp_telem_cfg)) {
			PIOS_Assert(0);
		}
		
		uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_TELEM_RF_RX_BUF_LEN);
		uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_TELEM_RF_TX_BUF_LEN);
		PIOS_Assert(rx_buffer);
		PIOS_Assert(tx_buffer);
		if (PIOS_COM_Init(&pios_com_telem_rf_id, &pios_udp_com_driver, pios_udp_telem_rf_id,
						  rx_buffer, PIOS_COM_TELEM_RF_RX_BUF_LEN,
						  tx_buffer, PIOS_COM_TELEM_RF_TX_BUF_LEN)) {
			PIOS_Assert(0);
		}
	}
#endif /* PIOS_INCLUDE_TELEMETRY_RF */


#if defined(PIOS_INCLUDE_GPS)
	{
		uintptr_t pios_tcp_gps_id;
		if (PIOS_TCP_Init(&pios_tcp_gps_id, &pios_tcp_gps_cfg)) {
			PIOS_Assert(0);
		}
		uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_GPS_RX_BUF_LEN);
		PIOS_Assert(rx_buffer);
		if (PIOS_COM_Init(&pios_com_gps_id, &pios_tcp_com_driver, pios_tcp_gps_id,
				  rx_buffer, PIOS_COM_GPS_RX_BUF_LEN,
				  NULL, 0)) {
			PIOS_Assert(0);
		}
	}
#endif	/* PIOS_INCLUDE_GPS */
#endif

#if defined(PIOS_INCLUDE_GCSRCVR)
	GCSReceiverInitialize();
	uintptr_t pios_gcsrcvr_id;
	PIOS_GCSRCVR_Init(&pios_gcsrcvr_id);
	uintptr_t pios_gcsrcvr_rcvr_id;
	if (PIOS_RCVR_Init(&pios_gcsrcvr_rcvr_id, &pios_gcsrcvr_rcvr_driver, pios_gcsrcvr_id)) {
		PIOS_Assert(0);
	}
	pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_GCS] = pios_gcsrcvr_rcvr_id;
#endif	/* PIOS_INCLUDE_GCSRCVR */

	// Register fake address.  Later if we really fake entire sensors then
	// it will make sense to have real queues registered.  For now if these
	// queues are used a crash is appropriate.
	PIOS_SENSORS_Register(PIOS_SENSOR_ACCEL, (xQueueHandle) 1);
	PIOS_SENSORS_Register(PIOS_SENSOR_GYRO, (xQueueHandle) 1);
	PIOS_SENSORS_Register(PIOS_SENSOR_MAG, (xQueueHandle) 1);
	PIOS_SENSORS_Register(PIOS_SENSOR_BARO, (xQueueHandle) 1);
}

/**
 * @}
 */
