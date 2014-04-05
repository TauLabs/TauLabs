/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup GSPModule GPS Module
 * @{ 
 *
 * @file       GPS.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      GPS module, handles UBX and NMEA streams from GPS
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
#include "GPS.h"

#include "gpsposition.h"
#include "airspeedactual.h"
#include "homelocation.h"
#include "gpstime.h"
#include "gpssatellites.h"
#include "gpsvelocity.h"
#include "modulesettings.h"

#include "NMEA.h"
#include "UBX.h"
#include "ubx_cfg.h"

#if defined(PIOS_GPS_PROVIDES_AIRSPEED)
#include "gps_airspeed.h"
#endif


// ****************
// Private functions

static void gpsTask(void *parameters);
static void updateSettings();

// ****************
// Private constants

#define GPS_TIMEOUT_MS                  500
#define GPS_COM_TIMEOUT_MS              100


#if defined(PIOS_GPS_MINIMAL)
	#define STACK_SIZE_BYTES            500
#else
	#define STACK_SIZE_BYTES            850
#endif // PIOS_GPS_MINIMAL

#define TASK_PRIORITY                   (tskIDLE_PRIORITY + 1)

// ****************
// Private variables

static uint32_t gpsPort;
static bool module_enabled = false;

static xTaskHandle gpsTaskHandle;

static char* gps_rx_buffer;

static uint32_t timeOfLastCommandMs;
static uint32_t timeOfLastUpdateMs;

static struct GPS_RX_STATS gpsRxStats;

// ****************
/**
 * Initialise the gps module
 * \return -1 if initialisation failed
 * \return 0 on success
 */

int32_t GPSStart(void)
{
	if (module_enabled) {
		if (gpsPort) {
			// Start gps task
			xTaskCreate(gpsTask, (signed char *)"GPS", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &gpsTaskHandle);
			TaskMonitorAdd(TASKINFO_RUNNING_GPS, gpsTaskHandle);
			return 0;
		}

		AlarmsSet(SYSTEMALARMS_ALARM_GPS, SYSTEMALARMS_ALARM_CRITICAL);
	}
	return -1;
}

/**
 * Initialise the gps module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t GPSInitialize(void)
{
	gpsPort = PIOS_COM_GPS;
	uint8_t	gpsProtocol;

#ifdef MODULE_GPS_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_GPS] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

#if defined(REVOLUTION)
	// These objects MUST be initialized for Revolution
	// because the rest of the system expects to just
	// attach to their queues
	GPSPositionInitialize();
	GPSVelocityInitialize();
	GPSTimeInitialize();
	GPSSatellitesInitialize();
	HomeLocationInitialize();
	UBloxInfoInitialize();
	updateSettings();

#else
	if (gpsPort && module_enabled) {
		GPSPositionInitialize();
		GPSVelocityInitialize();
#if !defined(PIOS_GPS_MINIMAL)
		GPSTimeInitialize();
		GPSSatellitesInitialize();
#endif
#if defined(PIOS_GPS_PROVIDES_AIRSPEED)
		AirspeedActualInitialize();
#endif
		updateSettings();
	}
#endif

	if (gpsPort && module_enabled) {
		ModuleSettingsGPSDataProtocolGet(&gpsProtocol);
		switch (gpsProtocol) {
			case MODULESETTINGS_GPSDATAPROTOCOL_NMEA:
				gps_rx_buffer = pvPortMalloc(NMEA_MAX_PACKET_LENGTH);
				break;
			case MODULESETTINGS_GPSDATAPROTOCOL_UBX:
				gps_rx_buffer = pvPortMalloc(sizeof(struct UBXPacket));
				break;
			default:
				gps_rx_buffer = NULL;
		}
		PIOS_Assert(gps_rx_buffer);

		return 0;
	}

	return -1;
}

MODULE_INITCALL(GPSInitialize, GPSStart);

// ****************
/**
 * Main gps task. It does not return.
 */

static void gpsTask(void *parameters)
{
	portTickType xDelay = MS2TICKS(GPS_COM_TIMEOUT_MS);
	uint32_t timeNowMs = TICKS2MS(xTaskGetTickCount());

	GPSPositionData gpsposition;
	uint8_t	gpsProtocol;

	ModuleSettingsGPSDataProtocolGet(&gpsProtocol);

#if defined(PIOS_GPS_PROVIDES_AIRSPEED)
	gps_airspeed_initialize();
#endif

	timeOfLastUpdateMs = timeNowMs;
	timeOfLastCommandMs = timeNowMs;


#if !defined(PIOS_GPS_MINIMAL)
	switch (gpsProtocol) {
#if defined(PIOS_INCLUDE_GPS_UBX_PARSER)
		case MODULESETTINGS_GPSDATAPROTOCOL_UBX:
		{
			uint8_t gpsAutoConfigure;
			ModuleSettingsGPSAutoConfigureGet(&gpsAutoConfigure);

			if (gpsAutoConfigure == MODULESETTINGS_GPSAUTOCONFIGURE_TRUE) {

				// Wait for power to stabilize before talking to external devices
				vTaskDelay(MS2TICKS(1000));

				// Runs through a number of possible GPS baud rates to
				// configure the ublox baud rate. This uses a NMEA string
				// so could work for either UBX or NMEA actually. This is
				// somewhat redundant with updateSettings below, but that
				// is only called on startup and is not an issue.
				ModuleSettingsGPSSpeedOptions baud_rate;
				ModuleSettingsGPSSpeedGet(&baud_rate);
				ubx_cfg_set_baudrate(gpsPort, baud_rate);

				vTaskDelay(MS2TICKS(1000));

				ubx_cfg_send_configuration(gpsPort, gps_rx_buffer);
			}
		}
			break;
#endif
	}
#endif /* PIOS_GPS_MINIMAL */

	GPSPositionGet(&gpsposition);
	// Loop forever
	while (1)
	{
		uint8_t c;

		// This blocks the task until there is something on the buffer
		while (PIOS_COM_ReceiveBuffer(gpsPort, &c, 1, xDelay) > 0)
		{
			int res;
			switch (gpsProtocol) {
#if defined(PIOS_INCLUDE_GPS_NMEA_PARSER)
				case MODULESETTINGS_GPSDATAPROTOCOL_NMEA:
					res = parse_nmea_stream (c,gps_rx_buffer, &gpsposition, &gpsRxStats);
					break;
#endif
#if defined(PIOS_INCLUDE_GPS_UBX_PARSER)
				case MODULESETTINGS_GPSDATAPROTOCOL_UBX:
					res = parse_ubx_stream (c,gps_rx_buffer, &gpsposition, &gpsRxStats);
					break;
#endif
				default:
					res = NO_PARSER; // this should not happen
					break;
			}

			if (res == PARSER_COMPLETE) {
				timeNowMs = TICKS2MS(xTaskGetTickCount());
				timeOfLastUpdateMs = timeNowMs;
				timeOfLastCommandMs = timeNowMs;
			}
		}

		// Check for GPS timeout
		timeNowMs = TICKS2MS(xTaskGetTickCount());
		if ((timeNowMs - timeOfLastUpdateMs) >= GPS_TIMEOUT_MS) {
			// we have not received any valid GPS sentences for a while.
			// either the GPS is not plugged in or a hardware problem or the GPS has locked up.
			uint8_t status = GPSPOSITION_STATUS_NOGPS;
			GPSPositionStatusSet(&status);
			AlarmsSet(SYSTEMALARMS_ALARM_GPS, SYSTEMALARMS_ALARM_ERROR);
		} else {
			// we appear to be receiving GPS sentences OK, we've had an update
			//criteria for GPS-OK taken from this post
			if (gpsposition.PDOP < 3.5f && 
			    gpsposition.Satellites >= 7 &&
			    (gpsposition.Status == GPSPOSITION_STATUS_FIX3D ||
			         gpsposition.Status == GPSPOSITION_STATUS_DIFF3D)) {
				AlarmsClear(SYSTEMALARMS_ALARM_GPS);
			} else if (gpsposition.Status == GPSPOSITION_STATUS_FIX3D ||
			           gpsposition.Status == GPSPOSITION_STATUS_DIFF3D)
						AlarmsSet(SYSTEMALARMS_ALARM_GPS, SYSTEMALARMS_ALARM_WARNING);
					else
						AlarmsSet(SYSTEMALARMS_ALARM_GPS, SYSTEMALARMS_ALARM_CRITICAL);
		}

	}
}


/**
 * Update the GPS settings, called on startup.
 */
static void updateSettings()
{
	if (gpsPort) {

		// Retrieve settings
		uint8_t speed;
		ModuleSettingsGPSSpeedGet(&speed);

		// Set port speed
		switch (speed) {
		case MODULESETTINGS_GPSSPEED_2400:
			PIOS_COM_ChangeBaud(gpsPort, 2400);
			break;
		case MODULESETTINGS_GPSSPEED_4800:
			PIOS_COM_ChangeBaud(gpsPort, 4800);
			break;
		case MODULESETTINGS_GPSSPEED_9600:
			PIOS_COM_ChangeBaud(gpsPort, 9600);
			break;
		case MODULESETTINGS_GPSSPEED_19200:
			PIOS_COM_ChangeBaud(gpsPort, 19200);
			break;
		case MODULESETTINGS_GPSSPEED_38400:
			PIOS_COM_ChangeBaud(gpsPort, 38400);
			break;
		case MODULESETTINGS_GPSSPEED_57600:
			PIOS_COM_ChangeBaud(gpsPort, 57600);
			break;
		case MODULESETTINGS_GPSSPEED_115200:
			PIOS_COM_ChangeBaud(gpsPort, 115200);
			break;
		case MODULESETTINGS_GPSSPEED_230400:
			PIOS_COM_ChangeBaud(gpsPort, 230400);
			break;
		}
	}
}

/** 
  * @}
  * @}
  */
