/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup PicoC Interpreter Module
 * @{ 
 *
 * @file       picoc_library.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      c-interpreter module for autonomous user programmed tasks
 *             library functions for uavo communication
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


// conditional compilation of the module
#include "pios.h"
#ifdef PIOS_INCLUDE_PICOC

#include "openpilot.h"
#include "picoc_port.h"

/**
 * string.h
 */
 #ifndef NO_STRING_FUNCTIONS
void LibStrcpy(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	char *To = (char *)Param[0]->Val->Pointer;
	char *From = (char *)Param[1]->Val->Pointer;

	while (*From != '\0')
		*To++ = *From++;

	*To = '\0';
}

void LibStrncpy(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	char *To = (char *)Param[0]->Val->Pointer;
	char *From = (char *)Param[1]->Val->Pointer;
	int Len = Param[2]->Val->Integer;

	for (; *From != '\0' && Len > 0; Len--)
		*To++ = *From++;

	if (Len > 0)
		*To = '\0';
}

void LibStrcmp(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	char *Str1 = (char *)Param[0]->Val->Pointer;
	char *Str2 = (char *)Param[1]->Val->Pointer;
	int StrEnded;

	for (StrEnded = FALSE; !StrEnded; StrEnded = (*Str1 == '\0' || *Str2 == '\0'), Str1++, Str2++)
	{
		if (*Str1 < *Str2) { ReturnValue->Val->Integer = -1; return; } 
		else if (*Str1 > *Str2) { ReturnValue->Val->Integer = 1; return; }
	}
	ReturnValue->Val->Integer = 0;
}

void LibStrncmp(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	char *Str1 = (char *)Param[0]->Val->Pointer;
	char *Str2 = (char *)Param[1]->Val->Pointer;
	int Len = Param[2]->Val->Integer;
	int StrEnded;

	for (StrEnded = FALSE; !StrEnded && Len > 0; StrEnded = (*Str1 == '\0' || *Str2 == '\0'), Str1++, Str2++, Len--)
	{
		if (*Str1 < *Str2) { ReturnValue->Val->Integer = -1; return; } 
		else if (*Str1 > *Str2) { ReturnValue->Val->Integer = 1; return; }
	}
	ReturnValue->Val->Integer = 0;
}

void LibStrcat(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	char *To = (char *)Param[0]->Val->Pointer;
	char *From = (char *)Param[1]->Val->Pointer;

	while (*To != '\0')
		To++;

	while (*From != '\0')
		*To++ = *From++;

	*To = '\0';
}

void LibIndex(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	char *Pos = (char *)Param[0]->Val->Pointer;
	int SearchChar = Param[1]->Val->Integer;

	while (*Pos != '\0' && *Pos != SearchChar)
		Pos++;

	if (*Pos != SearchChar)
		ReturnValue->Val->Pointer = NULL;
	else
		ReturnValue->Val->Pointer = Pos;
}

void LibRindex(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	char *Pos = (char *)Param[0]->Val->Pointer;
	int SearchChar = Param[1]->Val->Integer;

	ReturnValue->Val->Pointer = NULL;
	for (; *Pos != '\0'; Pos++)
	{
		if (*Pos == SearchChar)
			ReturnValue->Val->Pointer = Pos;
	}
}

void LibStrlen(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	char *Pos = (char *)Param[0]->Val->Pointer;
	int Len;

	for (Len = 0; *Pos != '\0'; Pos++)
		Len++;

	 ReturnValue->Val->Integer = Len;
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_string[] =
{
	{ LibStrcpy,	"void strcpy(char *,char *);" },
	{ LibStrncpy,	"void strncpy(char *,char *,int);" },
	{ LibStrcmp,	"int strcmp(char *,char *);" },
	{ LibStrncmp,	"int strncmp(char *,char *,int);" },
	{ LibStrcat,	"void strcat(char *,char *);" },
	{ LibIndex,		"char *index(char *,int);" },
	{ LibRindex,	"char *rindex(char *,int);" },
	{ LibStrlen,	"int strlen(char *);" },
	{ NULL, NULL }
};
#endif


/**
 * math.h
 */
#ifndef NO_FP
void LibSin(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = sin(Param[0]->Val->FP);
}

void LibCos(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = cos(Param[0]->Val->FP);
}

void LibTan(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = tan(Param[0]->Val->FP);
}

void LibAsin(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = asin(Param[0]->Val->FP);
}

void LibAcos(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = acos(Param[0]->Val->FP);
}

void LibAtan(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = atan(Param[0]->Val->FP);
}

void LibSinh(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = sinh(Param[0]->Val->FP);
}

void LibCosh(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = cosh(Param[0]->Val->FP);
}

void LibTanh(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = tanh(Param[0]->Val->FP);
}

void LibExp(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = exp(Param[0]->Val->FP);
}

void LibFabs(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = fabs(Param[0]->Val->FP);
}

void LibLog(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = log(Param[0]->Val->FP);
}

void LibLog10(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = log10(Param[0]->Val->FP);
}

void LibPow(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = pow(Param[0]->Val->FP, Param[1]->Val->FP);
}

void LibSqrt(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = sqrt(Param[0]->Val->FP);
}

void LibRound(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	/* this awkward definition of "round()" due to it being inconsistently declared in math.h */
	ReturnValue->Val->FP = ceil(Param[0]->Val->FP - 0.5);
}

void LibCeil(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = ceil(Param[0]->Val->FP);
}

void LibFloor(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->FP = floor(Param[0]->Val->FP);
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_math[] =
{
	{ LibSin,		"float sin(float);" },
	{ LibCos,		"float cos(float);" },
	{ LibTan,		"float tan(float);" },
	{ LibAsin,		"float asin(float);" },
	{ LibAcos,		"float acos(float);" },
	{ LibAtan,		"float atan(float);" },
	{ LibSinh,		"float sinh(float);" },
	{ LibCosh,		"float cosh(float);" },
	{ LibTanh,		"float tanh(float);" },
	{ LibExp,		"float exp(float);" },
	{ LibFabs,		"float fabs(float);" },
	{ LibLog,		"float log(float);" },
	{ LibLog10,		"float log10(float);" },
	{ LibPow,		"float pow(float,float);" },
	{ LibSqrt,		"float sqrt(float);" },
	{ LibRound,		"float round(float);" },
	{ LibCeil,		"float ceil(float);" },
	{ LibFloor,		"float floor(float);" },
	{ NULL, NULL }
};

/* some constants */
static double M_PIValue =	3.14159265358979323846;	/* pi */
static double M_EValue =	2.7182818284590452354;	/* e */

/* this is called when the header file is included */
void PlatformLibrarySetup_math(Picoc *pc)
{
	VariableDefinePlatformVar(pc, NULL, "M_PI", &pc->FPType, (union AnyValue *)&M_PIValue, FALSE);
	VariableDefinePlatformVar(pc, NULL, "M_E", &pc->FPType, (union AnyValue *)&M_EValue, FALSE);
}
#endif


/**
 * system.h
 */
#include "picocstatus.h"
#include "flightstatus.h"
#include "accessorydesired.h"
#include "manualcontrolsettings.h"

/* void delay(int): sleep for given ms-value */
void SystemDelay(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Integer > 0) {
		vTaskDelay(MS2TICKS(Param[0]->Val->Integer));
	}
}

/* void sync(int): synchronize an interval by given ms-value */
void SystemSync(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	static portTickType lastSysTime;
	if ((lastSysTime == 0) || (Param[0]->Val->Integer == 0)) {
		lastSysTime = xTaskGetTickCount();
	}
	if (Param[0]->Val->Integer > 0) {
		vTaskDelayUntil(&lastSysTime, MS2TICKS(Param[0]->Val->Integer));
	}
}

/* int armed(): returns armed status */
void SystemArmed(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	FlightStatusData data;
	FlightStatusArmedGet(&data.Armed);
	ReturnValue->Val->Integer = (data.Armed == FLIGHTSTATUS_ARMED_ARMED);
}

#ifdef PIOS_COM_PICOC
/* void ChangeBaud(long): changes the speed of picoc serial port */
void SystemChangeBaud(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if ((PIOS_COM_PICOC) && (Param[0]->Val->LongInteger > 0) && (Param[0]->Val->LongInteger <=115200)) {
		PIOS_COM_ChangeBaud(PIOS_COM_PICOC, Param[0]->Val->LongInteger);
	}
}
#endif

/* int TestValGet(): get the PicoCStatusTestValue */
void SystemTestValGet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	PicoCStatusData data;
	PicoCStatusTestValueGet(&data.TestValue);
	ReturnValue->Val->Integer = data.TestValue;
}

/* void TestValueSet(int): set the PicoCStatusTestValue */
void SystemTestValSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	PicoCStatusData data;
	data.TestValue = Param[0]->Val->Integer;
	PicoCStatusTestValueSet(&data.TestValue);
}

#ifndef NO_FP
/* float AccessoryValueGet(int): get the AccessoryDesiredAccessoryVal of the selected instance */
void SystemAccessoryValGet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	AccessoryDesiredData data;
	ReturnValue->Val->FP = (AccessoryDesiredInstGet(Param[0]->Val->Integer, &data) == 0) ? (float)data.AccessoryVal : 0;
}

/* AccessoryValueGet(float): set the AccessoryDesiredAccessoryVal of the selected instance */
void SystemAccessoryValSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	AccessoryDesiredData data;
	data.AccessoryVal = (float)Param[1]->Val->FP;
	AccessoryDesiredInstSet(Param[0]->Val->Integer, &data);
}
#endif

/* TxChannelValGet(int): get a tx value of the selected channel*/
void SystemTxChannelValGet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	extern uintptr_t pios_rcvr_group_map[];
	ManualControlSettingsData data;
	ManualControlSettingsChannelGroupsGet(data.ChannelGroups);
	ReturnValue->Val->Integer = PIOS_RCVR_Read(pios_rcvr_group_map[data.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_THROTTLE]], Param[0]->Val->Integer);
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_system[] =
{
	{ SystemDelay,			"void delay(int);" },
	{ SystemSync,			"void sync(int);" },
	{ SystemArmed,			"int armed();" },
#ifdef PIOS_COM_PICOC
	{ SystemChangeBaud,		"void ChangeBaud(long);" },
#endif
	{ SystemTestValGet,		"int TestValGet();" },
	{ SystemTestValSet,		"void TestValSet(int);" },
#ifndef NO_FP
	{ SystemAccessoryValGet,"float AccessoryValGet(int);" },
	{ SystemAccessoryValSet,"void AccessoryValSet(int,float);" },
#endif
	{ SystemTxChannelValGet,"int TxChannelValGet(int);" },
	{ NULL, NULL }
};


/**
 * attitudeactual.h
 */
#include "attitudeactual.h"

/* library functions */
#ifndef NO_FP
void AttitudeActualRoll(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	AttitudeActualData data;
	AttitudeActualGet(&data);
	ReturnValue->Val->FP = (double)data.Roll;
}

void AttitudeActualPitch(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	AttitudeActualData data;
	AttitudeActualGet(&data);
	ReturnValue->Val->FP = (double)data.Pitch;
}

void AttitudeActualYaw(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	AttitudeActualData data;
	AttitudeActualGet(&data);
	ReturnValue->Val->FP = (double)data.Yaw;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_attitudeactual[] =
{
#ifndef NO_FP
	{ AttitudeActualRoll,	"float AttitudeActualRoll();" },
	{ AttitudeActualPitch,	"float AttitudeActualPitch();" },
	{ AttitudeActualYaw,	"float AttitudeActualYaw();" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_attitudeactual(Picoc *pc)
{
	if (AttitudeActualHandle() == NULL)
		ProgramFailNoParser(pc, "no attitudeactual");
}


/**
 * baroaltitude.h
 */
#include "baroaltitude.h"

/* library functions */
#ifndef NO_FP
void BaroAltitudeAltitude(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	BaroAltitudeData data;
	BaroAltitudeGet(&data);
	ReturnValue->Val->FP = (double)data.Altitude;
}

void BaroAltitudeTemperature(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	BaroAltitudeData data;
	BaroAltitudeGet(&data);
	ReturnValue->Val->FP = (double)data.Temperature;
}

void BaroAltitudePressure(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	BaroAltitudeData data;
	BaroAltitudeGet(&data);
	ReturnValue->Val->FP = (double)data.Pressure;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_baroaltitude[] =
{
#ifndef NO_FP
	{ BaroAltitudeAltitude,		"float BaroAltitudeAltitude();" },
	{ BaroAltitudeTemperature,	"float BaroAltitudeTemperature();" },
	{ BaroAltitudePressure,		"float BaroAltitudePressure();" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_baroaltitude(Picoc *pc)
{
	if (BaroAltitudeHandle() == NULL)
		ProgramFailNoParser(pc, "no baroaltitude");
}


/**
 * flightbatterystate.h
 */
#include "flightbatterystate.h"

/* library functions */
#ifndef NO_FP
void FlightBatteryStateVoltage(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	FlightBatteryStateData data;
	FlightBatteryStateGet(&data);
	ReturnValue->Val->FP = (double)data.Voltage;
}

void FlightBatteryStateCurrent(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	FlightBatteryStateData data;
	FlightBatteryStateGet(&data);
	ReturnValue->Val->FP = (double)data.Current;
}

void FlightBatteryStateConsumedEnergy(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	FlightBatteryStateData data;
	FlightBatteryStateGet(&data);
	ReturnValue->Val->FP = (double)data.ConsumedEnergy;
}

void FlightBatteryStateEstimatedFlightTime(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	FlightBatteryStateData data;
	FlightBatteryStateGet(&data);
	ReturnValue->Val->FP = (double)data.EstimatedFlightTime;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_flightbatterystate[] =
{
#ifndef NO_FP
	{ FlightBatteryStateVoltage,				"float FlightBatteryStateVoltage();" },
	{ FlightBatteryStateCurrent,				"float FlightBatteryStateCurrent();" },
	{ FlightBatteryStateConsumedEnergy,			"float FlightBatteryStateConsumedEnergy();" },
	{ FlightBatteryStateEstimatedFlightTime,	"float FlightBatteryStateEstimatedFlightTime();" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_flightbatterystate(Picoc *pc)
{
	if (FlightBatteryStateHandle() == NULL)
		ProgramFailNoParser(pc, "no flightbatterystate");
}


/**
 * flightstatus.h
 */
#include "flightstatus.h"

/* library functions */
void FlightStatusArmed(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	FlightStatusData data;
	FlightStatusGet(&data);
	ReturnValue->Val->Integer = data.Armed;
}

void FlightStatusFlightMode(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	FlightStatusData data;
	FlightStatusGet(&data);
	ReturnValue->Val->Integer = data.FlightMode;
}

void FlightStatusControlSource(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	FlightStatusData data;
	FlightStatusGet(&data);
	ReturnValue->Val->Integer = data.ControlSource;
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_flightstatus[] =
{
	{ FlightStatusArmed,			"int FlightStatusArmed();" },
	{ FlightStatusFlightMode,		"int FlightStatusFlightMode();" },
	{ FlightStatusControlSource,	"int FlightStatusControlSource();" },
	{ NULL, NULL }
};


/**
 * gpsposition.h
 */
#include "gpsposition.h"

/* library functions */
void GPSPositionLatitude(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	GPSPositionData data;
	GPSPositionGet(&data);
	ReturnValue->Val->LongInteger = data.Latitude;
}

void GPSPositionLongitude(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	GPSPositionData data;
	GPSPositionGet(&data);
	ReturnValue->Val->LongInteger = data.Longitude;
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_gpsposition[] =
{
	{ GPSPositionLatitude,		"long GPSPositionLatitude();" },
	{ GPSPositionLongitude,		"long GPSPositionLongitude();" },
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_gpsposition(Picoc *pc)
{
	if (GPSPositionHandle() == NULL)
		ProgramFailNoParser(pc, "no gpsposition");
}


/**
 * pwm.h
 */
#include "actuatorsettings.h"
#include "manualcontrolsettings.h"
#include "pios_rcvr.h"

/* library functions */
void PWMFreqSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if ((Param[0]->Val->Integer > 0) && (Param[0]->Val->Integer <= ACTUATORSETTINGS_CHANNELUPDATEFREQ_NUMELEM)) {
		ActuatorSettingsData data;
		ActuatorSettingsChannelUpdateFreqGet(data.ChannelUpdateFreq);
		data.ChannelUpdateFreq[Param[0]->Val->Integer - 1] = Param[1]->Val->UnsignedInteger;
		ActuatorSettingsChannelUpdateFreqSet(data.ChannelUpdateFreq);
	}
}

void PWMMinSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if ((Param[0]->Val->Integer > 0) && (Param[0]->Val->Integer <= ACTUATORSETTINGS_CHANNELMIN_NUMELEM)) {
		ActuatorSettingsData data;
		ActuatorSettingsChannelMinGet(data.ChannelMin);
		data.ChannelMin[Param[0]->Val->Integer - 1] = Param[1]->Val->Integer;
		ActuatorSettingsChannelMinSet(data.ChannelMin);
	}
}

void PWMMaxSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if ((Param[0]->Val->Integer > 0) && (Param[0]->Val->Integer <= ACTUATORSETTINGS_CHANNELMAX_NUMELEM)) {
		ActuatorSettingsData data;
		ActuatorSettingsChannelMaxGet(data.ChannelMax);
		data.ChannelMax[Param[0]->Val->Integer - 1] = Param[1]->Val->Integer;
		ActuatorSettingsChannelMaxSet(data.ChannelMax);
	}
}

void PWMValSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if ((Param[0]->Val->Integer > 0) && (Param[0]->Val->Integer <= ACTUATORSETTINGS_CHANNELNEUTRAL_NUMELEM)) {
		ActuatorSettingsData data;
		ActuatorSettingsChannelNeutralGet(data.ChannelNeutral);
		data.ChannelNeutral[Param[0]->Val->Integer - 1] = Param[1]->Val->Integer;
		ActuatorSettingsChannelNeutralSet(data.ChannelNeutral);
	}
}

#ifdef PIOS_INCLUDE_PWM
void PWMValGet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	extern uintptr_t pios_rcvr_group_map[];
	ReturnValue->Val->Integer = PIOS_RCVR_Read(pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PWM], Param[0]->Val->Integer);
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_pwm[] =
{
	{ PWMFreqSet,		"void PWMFreqSet(int,unsigned int);" },
	{ PWMMinSet,		"void PWMMinSet(int,int);" },
	{ PWMMaxSet,		"void PWMMaxSet(int,int);" },
	{ PWMValSet,		"void PWMValSet(int,int);" },
#ifdef PIOS_INCLUDE_PWM
	{ PWMValGet,		"int PWMValGet(int);" },
#endif
	{ NULL, NULL }
};


/* list all includes */
void PlatformLibraryInit(Picoc *pc)
{
#ifndef NO_STRING_FUNCTIONS
	IncludeRegister(pc, "string.h", NULL, &PlatformLibrary_string[0], NULL);
#endif
#ifndef NO_FP
	IncludeRegister(pc, "math.h", &PlatformLibrarySetup_math, &PlatformLibrary_math[0], NULL);
#endif
	IncludeRegister(pc, "system.h", NULL, &PlatformLibrary_system[0], NULL);
	IncludeRegister(pc, "attitudeactual.h", &PlatformLibrarySetup_attitudeactual, &PlatformLibrary_attitudeactual[0], NULL);
	IncludeRegister(pc, "baroaltitude.h", &PlatformLibrarySetup_baroaltitude, &PlatformLibrary_baroaltitude[0], NULL);
	IncludeRegister(pc, "flightbatterystate.h", &PlatformLibrarySetup_flightbatterystate, &PlatformLibrary_flightbatterystate[0], NULL);
	IncludeRegister(pc, "flightstatus.h", NULL, &PlatformLibrary_flightstatus[0], NULL);
	IncludeRegister(pc, "gpsposition.h", &PlatformLibrarySetup_gpsposition, &PlatformLibrary_gpsposition[0], NULL);
	IncludeRegister(pc, "pwm.h", NULL, &PlatformLibrary_pwm[0], NULL);
}

#endif /* PIOS_INCLUDE_PICOC */

/**
 * @}
 * @}
 */
