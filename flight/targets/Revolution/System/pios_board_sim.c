/**
 ******************************************************************************
 *
 * @file       plop_board.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      Defines board specific static initializers for hardware for the OpenPilot board.
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

#include <plop.h>
#include "plop_board_sim.h"
#include <plop_com_priv.h>
#include <plop_tcp_priv.h>
#include <plop_udp_priv.h>
#include <openpilot.h>
#include <uavobjectsinit.h>

#include "accels.h"
#include "baroaltitude.h"
#include "gpsposition.h"
#include "gyros.h"
#include "gyrosbias.h"
#include "magnetometer.h"
#include "manualcontrolsettings.h"

#include "plop_rcvr_priv.h"
#include "plop_gcsrcvr_priv.h"

void Stack_Change() {
}

void Stack_Change_Weak() {
}


const struct plop_tcp_cfg plop_tcp_telem_cfg = {
  .ip = "0.0.0.0",
  .port = 9000,
};

const struct plop_udp_cfg plop_udp_telem_cfg = {
	.ip = "0.0.0.0",
	.port = 9000,
};

const struct plop_tcp_cfg plop_tcp_gps_cfg = {
  .ip = "0.0.0.0",
  .port = 9001,
};
const struct plop_tcp_cfg plop_tcp_debug_cfg = {
  .ip = "0.0.0.0",
  .port = 9002,
};

#ifdef plop_COM_AUX
/*
 * AUX USART
 */
const struct plop_tcp_cfg plop_tcp_aux_cfg = {
  .ip = "0.0.0.0",
  .port = 9003,
};
#endif

#define plop_COM_TELEM_RF_RX_BUF_LEN 192
#define plop_COM_TELEM_RF_TX_BUF_LEN 192
#define plop_COM_GPS_RX_BUF_LEN 96

/**
 * Simulation of the flash filesystem
 */
#include "../../../tests/logfs/plop_flash_ut_priv.h"
const struct plop_flash_ut_cfg flash_config = {
	.size_of_flash  = 0x00300000,
	.size_of_sector = 0x00010000,
};

#include "plop_flashfs_logfs_priv.h"

const struct flashfs_logfs_cfg flashfs_config_partition_a = {
	.fs_magic      = 0x89abceef,
	.total_fs_size = 0x00200000, /* 2M bytes (32 sectors) */
	.arena_size    = 0x00010000, /* 256 * slot size */
	.slot_size     = 0x00000100, /* 256 bytes */

	.start_offset  = 0,	     /* start at the beginning of the chip */
	.sector_size   = 0x00010000, /* 64K bytes */
	.page_size     = 0x00000100, /* 256 bytes */
};

const struct flashfs_logfs_cfg flashfs_config_partition_b = {
	.fs_magic      = 0x89abceef,
	.total_fs_size = 0x00100000, /* 1M bytes (16 sectors) */
	.arena_size    = 0x00010000, /* 64 * slot size */
	.slot_size     = 0x00000400, /* 256 bytes */

	.start_offset  = 0x00200000, /* start after partition a */
	.sector_size   = 0x00010000, /* 64K bytes */
	.page_size     = 0x00000100, /* 256 bytes */
};

uintptr_t plop_uavo_settings_fs_id;
uintptr_t plop_waypoints_settings_fs_id;

/*
 * Board specific number of devices.
 */
extern const struct plop_com_driver plop_serial_com_driver;
extern const struct plop_com_driver plop_udp_com_driver;
extern const struct plop_com_driver plop_tcp_com_driver;

uint32_t plop_com_telem_rf_id;
uint32_t plop_com_telem_usb_id;
uint32_t plop_com_gps_id;
uint32_t plop_com_aux_id;
uint32_t plop_com_spectrum_id;
uint32_t plop_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE];

/**
 * plop_Board_Init()
 * initializes all the core systems on this specific hardware
 * called from System/openpilot.c
 */
void plop_Board_Init(void) {

	/* Delay system */
	plop_DELAY_Init();

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

	uintptr_t flash_id;
	int32_t retval = plop_Flash_UT_Init(&flash_id, &flash_config);
  	if (retval != 0)
		fprintf(stderr, "Unable to initialize flash ut simulator: %d\n", retval);

  	if(plop_FLASHFS_Logfs_Init(&plop_waypoints_settings_fs_id, &flashfs_config_partition_b, &plop_ut_flash_driver, flash_id) != 0)
		fprintf(stderr, "Unable to open the waypoints partition\n");


#if defined(plop_INCLUDE_COM)
#if defined(plop_INCLUDE_TELEMETRY_RF) && 1
	{
		uint32_t plop_tcp_telem_rf_id;
		if (plop_TCP_Init(&plop_tcp_telem_rf_id, &plop_tcp_telem_cfg)) {
			plop_Assert(0);
		}

		uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(plop_COM_TELEM_RF_RX_BUF_LEN);
		uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(plop_COM_TELEM_RF_TX_BUF_LEN);
		plop_Assert(rx_buffer);
		plop_Assert(tx_buffer);
		if (plop_COM_Init(&plop_com_telem_rf_id, &plop_tcp_com_driver, plop_tcp_telem_rf_id,
						  rx_buffer, plop_COM_TELEM_RF_RX_BUF_LEN,
						  tx_buffer, plop_COM_TELEM_RF_TX_BUF_LEN)) {
			plop_Assert(0);
		}
	}
#endif /* plop_INCLUDE_TELEMETRY_RF */

#if defined(plop_INCLUDE_TELEMETRY_RF) && 0
	{
		uint32_t plop_udp_telem_rf_id;
		if (plop_UDP_Init(&plop_udp_telem_rf_id, &plop_udp_telem_cfg)) {
			plop_Assert(0);
		}
		
		uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(plop_COM_TELEM_RF_RX_BUF_LEN);
		uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(plop_COM_TELEM_RF_TX_BUF_LEN);
		plop_Assert(rx_buffer);
		plop_Assert(tx_buffer);
		if (plop_COM_Init(&plop_com_telem_rf_id, &plop_udp_com_driver, plop_udp_telem_rf_id,
						  rx_buffer, plop_COM_TELEM_RF_RX_BUF_LEN,
						  tx_buffer, plop_COM_TELEM_RF_TX_BUF_LEN)) {
			plop_Assert(0);
		}
	}
#endif /* plop_INCLUDE_TELEMETRY_RF */


#if defined(plop_INCLUDE_GPS)
	{
		uint32_t plop_tcp_gps_id;
		if (plop_TCP_Init(&plop_tcp_gps_id, &plop_tcp_gps_cfg)) {
			plop_Assert(0);
		}
		uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(plop_COM_GPS_RX_BUF_LEN);
		plop_Assert(rx_buffer);
		if (plop_COM_Init(&plop_com_gps_id, &plop_tcp_com_driver, plop_tcp_gps_id,
				  rx_buffer, plop_COM_GPS_RX_BUF_LEN,
				  NULL, 0)) {
			plop_Assert(0);
		}
	}
#endif	/* plop_INCLUDE_GPS */
#endif

#if defined(plop_INCLUDE_GCSRCVR)
	GCSReceiverInitialize();
	uint32_t plop_gcsrcvr_id;
	plop_GCSRCVR_Init(&plop_gcsrcvr_id);
	uint32_t plop_gcsrcvr_rcvr_id;
	if (plop_RCVR_Init(&plop_gcsrcvr_rcvr_id, &plop_gcsrcvr_rcvr_driver, plop_gcsrcvr_id)) {
		plop_Assert(0);
	}
	plop_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_GCS] = plop_gcsrcvr_rcvr_id;
#endif	/* plop_INCLUDE_GCSRCVR */

	// Register fake address.  Later if we really fake entire sensors then
	// it will make sense to have real queues registered.  For now if these
	// queues are used a crash is appropriate.
	plop_SENSORS_Register(plop_SENSOR_ACCEL, (xQueueHandle) 1);
	plop_SENSORS_Register(plop_SENSOR_GYRO, (xQueueHandle) 1);
	plop_SENSORS_Register(plop_SENSOR_MAG, (xQueueHandle) 1);
	plop_SENSORS_Register(plop_SENSOR_BARO, (xQueueHandle) 1);
}

/**
 * @}
 */
