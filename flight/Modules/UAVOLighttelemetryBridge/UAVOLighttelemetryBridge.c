/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup UAVOLighttelemetryBridge UAVO to Lighttelemetry Bridge Module
 * @{ 
 *
 * @file       UAVOLighttelemetryBridge.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Bridges selected UAVObjects to a minimal one way telemetry 
 *			   protocol for really low bitrates (<1000 bits/s). This can be 
 *			   used with FSK audio modems or increase range for serial telemetry.
 *			   Useful for ground osd & antenna tracker with long range.
 *			   Will be extended with different frames in the future.
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

#if defined(PIOS_INCLUDE_LIGHTTELEMETRY)

#include "openpilot.h"
#include "modulesettings.h"
#include "gpsposition.h"
#include "baroaltitude.h"

// Private constants
#define STACK_SIZE_BYTES 512
#define TASK_PRIORITY (tskIDLE_PRIORITY+1)
#define UPDATE_PERIOD 200

// Private types

// Private variables
static xTaskHandle taskHandle;

static uint32_t lighttelemetryPort;

// Private functions
static void uavoLighttelemetryBridgeTask(void *parameters);
static void updateSettings();

void SendData(int32_t Lat,int32_t Lon,uint8_t Speed,int32_t Alt,int8_t Sats, int8_t Fix);

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t uavoLighttelemetryBridgeStart()
{

	xTaskCreate(uavoLighttelemetryBridgeTask, (signed char *)"uavoLighttelemetryBridge", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_UAVOLIGHTTELEMETRYBRIDGE, taskHandle);
	return 0;
}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t uavoLighttelemetryBridgeInitialize()
{

	// Update telemetry settings
	lighttelemetryPort = PIOS_COM_LIGHTTELEMETRY;
	updateSettings();

	return 0;
}
MODULE_INITCALL(uavoLighttelemetryBridgeInitialize, uavoLighttelemetryBridgeStart);

/**
 * Module thread, should not return.
 */
static void uavoLighttelemetryBridgeTask(void *parameters)
{
	int32_t lt_latitude = 0;
	int32_t lt_longitude = 0;
    uint8_t lt_groundspeed = 0;
	int32_t lt_altitude = 0;
	uint8_t lt_gpsfix = 0;
	uint8_t lt_gpssats = 0;
	portTickType lastSysTime;
	GPSPositionData pdata;
	BaroAltitudeData bdata;
	
	GPSPositionInitialize();
	BaroAltitudeInitialize();
	// Main task loop
	lastSysTime = xTaskGetTickCount();
	while (1)
	{
		GPSPositionGet(&pdata);

		lt_latitude=pdata.Latitude;
		lt_longitude=pdata.Longitude;
		lt_groundspeed = (uint8_t)round(pdata.Groundspeed); //rounded m/s .

		if (BaroAltitudeHandle() != NULL) {
			BaroAltitudeGet(&bdata);
			lt_altitude = (int32_t)round(bdata.Altitude * 100); //Baro alt in cm.
		}
		else if (GPSPositionHandle() != NULL) lt_altitude = (int32_t)round(pdata.Altitude * 100); //GPS alt in cm.
		
		switch (pdata.Status) {
			case GPSPOSITION_STATUS_NOGPS:
				lt_gpsfix = 0;
				break;
			case GPSPOSITION_STATUS_NOFIX:
				lt_gpsfix = 1;
				break;
			case GPSPOSITION_STATUS_FIX2D:
				lt_gpsfix = 2;
				break;
			case GPSPOSITION_STATUS_FIX3D:
				lt_gpsfix = 3;
				break;
			default:
				lt_gpsfix = 0;
				break;
		}
		
		lt_gpssats = (int8_t)pdata.Satellites;
			
		SendData(lt_latitude,lt_longitude,lt_groundspeed,lt_altitude,lt_gpssats,lt_gpsfix);
		// Delay until it is time to read the next sample
		vTaskDelayUntil(&lastSysTime, UPDATE_PERIOD / portTICK_RATE_MS);
	}
}


void SendData(int32_t Lat,int32_t Lon,uint8_t Speed,int32_t Alt,int8_t Sats, int8_t Fix)
{
	uint32_t outputPort;
	uint8_t LTBuff[18];
	



	//protocol: START(2 bytes)FRAMEID(1byte)LAT(cm,4 bytes)LON(cm,4bytes)SPEED(m/s,1bytes)ALT(cm,4bytes)SATS(6bits)FIX(2bits)CRC(xor,1byte)
	//START
	LTBuff[0]=0x24; //$
	LTBuff[1]=0x54; //T
	//FRAMEID
	LTBuff[2]=0x47; // G ( gps frame at 5hz )
    //PAYLOAD
	LTBuff[3]=(Lat >> 8*0) & 0xFF;
	LTBuff[4]=(Lat >> 8*1) & 0xFF;
	LTBuff[5]=(Lat >> 8*2) & 0xFF;
	LTBuff[6]=(Lat >> 8*3) & 0xFF;
	LTBuff[7]=(Lon >> 8*0) & 0xFF;
	LTBuff[8]=(Lon >> 8*1) & 0xFF;
	LTBuff[9]=(Lon >> 8*2) & 0xFF;
	LTBuff[10]=(Lon >> 8*3) & 0xFF;	
	LTBuff[11]=(Speed >> 8*0) & 0xFF;
	LTBuff[12]=(Alt >> 8*0) & 0xFF;
	LTBuff[13]=(Alt >> 8*1) & 0xFF;
	LTBuff[14]=(Alt >> 8*2) & 0xFF;
	LTBuff[15]=(Alt >> 8*3) & 0xFF;
	LTBuff[16]= ((Sats << 2)& 0xFF ) | (Fix & 0b00000011) ; // last 6 bits: sats number, first 2:fix type (0,1,2,3)

	//CRC
	uint8_t LTCrc = 0x00;
	for (int i = 3; i < 17; i++) {
          LTCrc ^= LTBuff[i];
    }
	LTBuff[17]=LTCrc;
	outputPort = lighttelemetryPort;
	if (outputPort) {
		PIOS_COM_SendBuffer(outputPort, LTBuff, 18);
	}
}

/**
 * Update the telemetry settings, called on startup.
 * FIXME: This should be in the TelemetrySettings object. But objects
 * have too much overhead yet. Also the telemetry has no any specific
 * settings, etc. Thus the ModuleSettings object which contains the
 * telemetry port speed is used for now.
 */
static void updateSettings()
{
	
	if (lighttelemetryPort) {
		// Retrieve settings
		uint8_t speed;
		ModuleSettingsLightTelemetrySpeedGet(&speed);

		// Set port speed
		switch (speed) {
		case MODULESETTINGS_LIGHTTELEMETRYSPEED_1200:
			PIOS_COM_ChangeBaud(lighttelemetryPort, 1200);
			break;
		case MODULESETTINGS_LIGHTTELEMETRYSPEED_2400:
			PIOS_COM_ChangeBaud(lighttelemetryPort, 2400);
			break;
		case MODULESETTINGS_LIGHTTELEMETRYSPEED_4800:
			PIOS_COM_ChangeBaud(lighttelemetryPort, 4800);
			break;
		case MODULESETTINGS_LIGHTTELEMETRYSPEED_9600:
			PIOS_COM_ChangeBaud(lighttelemetryPort, 9600);
			break;
		case MODULESETTINGS_LIGHTTELEMETRYSPEED_19200:
			PIOS_COM_ChangeBaud(lighttelemetryPort, 19200);
			break;
		case MODULESETTINGS_LIGHTTELEMETRYSPEED_38400:
			PIOS_COM_ChangeBaud(lighttelemetryPort, 38400);
			break;
		case MODULESETTINGS_LIGHTTELEMETRYSPEED_57600:
			PIOS_COM_ChangeBaud(lighttelemetryPort, 57600);
			break;
		case MODULESETTINGS_LIGHTTELEMETRYSPEED_115200:
			PIOS_COM_ChangeBaud(lighttelemetryPort, 115200);
			break;
		}
	}
}

#endif //end define lighttelemetry
/**
 * @}
 * @}
 */
