/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup UAVOLighttelemetryBridge UAVO to Lighttelemetry Bridge Module
 * @{ 
 *
 * @file	   UAVOLighttelemetryBridge.c
 * @author	   Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief	   Bridges selected UAVObjects to a minimal one way telemetry 
 *			   protocol for really low bitrates (1200/2400 bauds). This can be 
 *			   used with FSK audio modems or increase range for serial telemetry.
 *			   Effective for ground OSD, groundstation HUD and Antenna tracker.
 *			   
 *				Protocol details: 3 different frames, little endian.
 *				  * G Frame (GPS position) (2hz @ 1200 bauds , 5hz >= 2400 bauds): 18BYTES
 *					0x24 0x54 0x47 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF	 0xFF	0xC0   
 *					 $	   T	G  --------LAT-------- -------LON---------	SPD --------ALT-------- SAT/FIX	 CRC
 *				  * A Frame (Attitude) (5hz @ 1200bauds , 10hz >= 2400bauds): 10BYTES
 *					0x24 0x54 0x41 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xC0	
 *					 $	   T   A   --PITCH-- --ROLL--- -HEADING-  CRC
 *				  * S Frame (Sensors) (2hz @ 1200bauds, 5hz >= 2400bauds): 11BYTES
 *					0x24 0x54 0x53 0xFF 0xFF  0xFF 0xFF	   0xFF	   0xFF		 0xFF		0xC0	 
 *					 $	   T   S   VBAT(mv)	 Current(ma)   RSSI	 AIRSPEED  ARM/FS/FMOD	 CRC
 *
 *
 * @see		   The GNU Public License (GPL) Version 3
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
#include "modulesettings.h"
#include "attitudeactual.h"
#include "gpsposition.h"
#include "baroaltitude.h"
#include "flightbatterysettings.h"
#include "flightbatterystate.h"
#include "gpsposition.h"
#include "airspeedactual.h"
#include "accels.h"
#include "manualcontrolcommand.h"
#include "flightstatus.h"

#if defined(PIOS_INCLUDE_LIGHTTELEMETRY)
// Private constants
#define STACK_SIZE_BYTES 512
#define TASK_PRIORITY (tskIDLE_PRIORITY+1)
#define UPDATE_PERIOD 100

#define LTM_GFRAME_SIZE 18
#define LTM_AFRAME_SIZE 10
#define LTM_SFRAME_SIZE 11

// Private types

// Private variables
static xTaskHandle taskHandle;

static uint32_t lighttelemetryPort;
static uint8_t ltm_scheduler;
static uint8_t ltm_slowrate;

// Private functions
static void uavoLighttelemetryBridgeTask(void *parameters);
static void updateSettings();

static void send_LTM_Packet(uint8_t *LTPacket, uint8_t LTPacket_size);
static void send_LTM_Gframe();
static void send_LTM_Aframe();
static void send_LTM_Sframe();

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
	ltm_scheduler = 1;
	updateSettings();
	uint8_t speed;
	ModuleSettingsLightTelemetrySpeedGet(&speed);
	if (speed == MODULESETTINGS_LIGHTTELEMETRYSPEED_1200)
		ltm_slowrate = 1;
	else 
		ltm_slowrate = 0;
	return 0;
}
MODULE_INITCALL(uavoLighttelemetryBridgeInitialize, uavoLighttelemetryBridgeStart);


/*#######################################################################
 * Module thread, should not return.
 *#######################################################################
*/

static void uavoLighttelemetryBridgeTask(void *parameters)
{
	portTickType lastSysTime;

	// Main task loop
	lastSysTime = xTaskGetTickCount();
	while (1)
	{

		if (ltm_scheduler & 1) {	// is odd
			send_LTM_Aframe();
		}
		else						// is even
		{
			if (ltm_slowrate == 0)
				send_LTM_Aframe();
				
			if (ltm_scheduler % 4 == 0)
				send_LTM_Sframe();
			else 
				send_LTM_Gframe();
		}
		ltm_scheduler++;
		if (ltm_scheduler > 10)
			ltm_scheduler = 1;
		// Delay until it is time to read the next sample
		vTaskDelayUntil(&lastSysTime, UPDATE_PERIOD / portTICK_RATE_MS);
	}
}

/*#######################################################################
 * Internal functions
 *#######################################################################
*/
//GPS packet
static void send_LTM_Gframe() 
{
	GPSPositionData pdata;
	BaroAltitudeData bdata;
	GPSPositionInitialize();
	BaroAltitudeInitialize();
	 //prepare data
	GPSPositionGet(&pdata);

	int32_t lt_latitude = pdata.Latitude;
	int32_t lt_longitude = pdata.Longitude;
	uint8_t lt_groundspeed = (uint8_t)roundf(pdata.Groundspeed); //rounded m/s .
	int32_t lt_altitude = 0;
	if (BaroAltitudeHandle() != NULL) {
		BaroAltitudeGet(&bdata);
		lt_altitude = (int32_t)roundf(bdata.Altitude * 100.0f); //Baro alt in cm.
	}
	else if (GPSPositionHandle() != NULL)
		lt_altitude = (int32_t)roundf(pdata.Altitude * 100.0f); //GPS alt in cm.
	
	uint8_t lt_gpsfix;
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
	
	uint8_t lt_gpssats = (int8_t)pdata.Satellites;
	//pack G frame	
	uint8_t LTBuff[LTM_GFRAME_SIZE];
	//G Frame: $T(2 bytes)G(1byte)LAT(cm,4 bytes)LON(cm,4bytes)SPEED(m/s,1bytes)ALT(cm,4bytes)SATS(6bits)FIX(2bits)CRC(xor,1byte)
	//START
	LTBuff[0]  = 0x24; //$
	LTBuff[1]  = 0x54; //T
	//FRAMEID
	LTBuff[2]  = 0x47; //G
	//PAYLOAD
	LTBuff[3]  = (lt_latitude >> 8*0) & 0xFF;
	LTBuff[4]  = (lt_latitude >> 8*1) & 0xFF;
	LTBuff[5]  = (lt_latitude >> 8*2) & 0xFF;
	LTBuff[6]  = (lt_latitude >> 8*3) & 0xFF;
	LTBuff[7]  = (lt_longitude >> 8*0) & 0xFF;
	LTBuff[8]  = (lt_longitude >> 8*1) & 0xFF;
	LTBuff[9]  = (lt_longitude >> 8*2) & 0xFF;
	LTBuff[10] = (lt_longitude >> 8*3) & 0xFF;	
	LTBuff[11] = (lt_groundspeed >> 8*0) & 0xFF;
	LTBuff[12] = (lt_altitude >> 8*0) & 0xFF;
	LTBuff[13] = (lt_altitude >> 8*1) & 0xFF;
	LTBuff[14] = (lt_altitude >> 8*2) & 0xFF;
	LTBuff[15] = (lt_altitude >> 8*3) & 0xFF;
	LTBuff[16] = ((lt_gpssats << 2)& 0xFF ) | (lt_gpsfix & 0b00000011) ; // last 6 bits: sats number, first 2:fix type (0,1,2,3)

	send_LTM_Packet(LTBuff,LTM_GFRAME_SIZE);
}

//Attitude packet
static void send_LTM_Aframe() 
{
	//prepare data
	AttitudeActualData adata;
	AttitudeActualGet(&adata);
	int16_t lt_pitch   = (int16_t)(roundf(adata.Pitch));	//-180/180°
	int16_t lt_roll	   = (int16_t)(roundf(adata.Roll));		//-180/180°
	int16_t lt_heading = (int16_t)(roundf(adata.Yaw));		//-180/180°
	//pack A frame	
	uint8_t LTBuff[LTM_AFRAME_SIZE];
	
	//A Frame: $T(2 bytes)A(1byte)PITCH(2 bytes)ROLL(2bytes)HEADING(2bytes)CRC(xor,1byte)
	//START
	LTBuff[0] = 0x24; //$
	LTBuff[1] = 0x54; //T
	//FRAMEID
	LTBuff[2] = 0x41; //A 
	//PAYLOAD
	LTBuff[3] = (lt_pitch >> 8*0) & 0xFF;
	LTBuff[4] = (lt_pitch >> 8*1) & 0xFF;
	LTBuff[5] = (lt_roll >> 8*0) & 0xFF;
	LTBuff[6] = (lt_roll >> 8*1) & 0xFF;
	LTBuff[7] = (lt_heading >> 8*0) & 0xFF;
	LTBuff[8] = (lt_heading >> 8*1) & 0xFF;
	send_LTM_Packet(LTBuff,LTM_AFRAME_SIZE);
}

//Sensors packet
static void send_LTM_Sframe() 
{
	//prepare data
	uint16_t lt_vbat = 0;
	uint16_t lt_amp = 0;
	uint8_t	 lt_rssi = 0;
	uint8_t	 lt_airspeed = 0;
	uint8_t	 lt_arm = 0;
	uint8_t	 lt_failsafe = 0;
	uint8_t	 lt_flightmode = 0;
	
	
	if (FlightBatteryStateHandle() != NULL) {
		FlightBatteryStateData sdata;
		FlightBatteryStateGet(&sdata);
		lt_vbat = (uint16_t)roundf(sdata.Voltage*1000);	  //Battery voltage in mv
		lt_amp = (uint16_t)roundf(sdata.ConsumedEnergy);	  //mA consumed
	}
	if (ManualControlCommandHandle() != NULL) {
		ManualControlCommandData mdata;
		ManualControlCommandGet(&mdata);
		lt_rssi = (uint8_t)mdata.Rssi;					  //RSSI in %
	}
	if (AirspeedActualHandle() != NULL) {
		AirspeedActualData adata;
		AirspeedActualGet (&adata);
		lt_airspeed = (uint8_t)roundf(adata.TrueAirspeed);	  //Airspeed in m/s
	}
	FlightStatusData fdata;
	FlightStatusGet(&fdata);
	lt_arm = fdata.Armed;									  //Armed status
	if (lt_arm == 1)		//arming , we don't use this one
		lt_arm = 0;		
	else if (lt_arm == 2)  // armed
		lt_arm = 1;
	if (fdata.ControlSource == FLIGHTSTATUS_CONTROLSOURCE_FAILSAFE)
		lt_failsafe = 1;
	else
		lt_failsafe = 0;
	
	// Flight mode(0-19): 0: Manual, 1: Rate, 2: Attitude/Angle, 3: Horizon, 4: Acro, 5: Stabilized1, 6: Stabilized2, 7: Stabilized3,
	// 8: Altitude Hold, 9: Loiter/GPS Hold, 10: Auto/Waypoints, 11: Heading Hold / headFree, 
	// 12: Circle, 13: RTH, 14: FollowMe, 15: LAND, 16:FlybyWireA, 17: FlybywireB, 18: Cruise, 19: Unknown

	switch (fdata.FlightMode) {
	case FLIGHTSTATUS_FLIGHTMODE_MANUAL:
		lt_flightmode = 0; break;
	case FLIGHTSTATUS_FLIGHTMODE_STABILIZED1:
		lt_flightmode = 5; break;
	case FLIGHTSTATUS_FLIGHTMODE_STABILIZED2:
		lt_flightmode = 6; break;
	case FLIGHTSTATUS_FLIGHTMODE_STABILIZED3:
		lt_flightmode = 7; break;
	case FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD:
		lt_flightmode = 8; break;
	case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
		lt_flightmode = 9; break;
	case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
		lt_flightmode = 13; break;
	case FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER:
		lt_flightmode = 10; break;
	default:
		lt_flightmode = 19; //Unknown
	}
	//pack A frame	
	uint8_t LTBuff[LTM_SFRAME_SIZE];
	
	//A Frame: $T(2 bytes)A(1byte)PITCH(2 bytes)ROLL(2bytes)HEADING(2bytes)CRC(xor,1byte)
	//START
	LTBuff[0] = 0x24; //$
	LTBuff[1] = 0x54; //T
	//FRAMEID
	LTBuff[2] = 0x53; //S 
	//PAYLOAD
	LTBuff[3] = (lt_vbat >> 8*0) & 0xFF;
	LTBuff[4] = (lt_vbat >> 8*1) & 0xFF;
	LTBuff[5] = (lt_amp >> 8*0) & 0xFF;
	LTBuff[6] = (lt_amp >> 8*1) & 0xFF;
	LTBuff[7] = (lt_rssi >> 8*0) & 0xFF;
	LTBuff[8] = (lt_airspeed >> 8*0) & 0xFF;
	LTBuff[9] = ((lt_flightmode << 2)& 0xFF ) | ((lt_failsafe << 1)& 0b00000010 ) | (lt_arm & 0b00000001) ; // last 6 bits: flight mode, 2nd bit: failsafe, 1st bit: Arm status.
	send_LTM_Packet(LTBuff,LTM_SFRAME_SIZE);
}

static void send_LTM_Packet(uint8_t *LTPacket, uint8_t LTPacket_size)
{
	//calculate Checksum
	uint8_t LTCrc = 0x00;
	for (int i = 3; i < LTPacket_size-1; i++) {
		LTCrc ^= LTPacket[i];
	}
	LTPacket[LTPacket_size-1] = LTCrc;
	if (lighttelemetryPort) {
		PIOS_COM_SendBuffer(lighttelemetryPort, LTPacket, LTPacket_size);
	}
}

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
