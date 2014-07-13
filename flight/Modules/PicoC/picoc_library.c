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
#include "flightstatus.h"

// Private variables
static int accesslevel;

/* check access level */
bool security(int neededlevel)
{

	if (accesslevel < neededlevel)
		// access level is insufficient 
		return false;

	FlightStatusData data;
	FlightStatusArmedGet(&data.Armed);

	// in level 1 flightstatus has also to be disarmed
	if ((accesslevel <= 1) && (data.Armed != FLIGHTSTATUS_ARMED_DISARMED))
		return false;

	// level 2 or higher is currently like root

	// all checks are ok
	return true;
}

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
#include "actuatorsettings.h"
#include "mixersettings.h"
#include "pios_rcvr.h"

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

/* void AccessLevelSet(int): sets the access level. Used for security in some library functions */
void SystemAccessLevelSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	accesslevel = Param[0]->Val->Integer;
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

/* AccessoryValueSet(int,float): set the AccessoryDesiredAccessoryVal of the selected instance */
void SystemAccessoryValSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	AccessoryDesiredData data;
	data.AccessoryVal = (float)Param[1]->Val->FP;
	AccessoryDesiredInstSet(Param[0]->Val->Integer, &data);
}
#endif

/* PWMFreqSet(int,uint): set output update speed */
void SystemPWMFreqSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (!security(1))
		return;
	if ((Param[0]->Val->Integer >= 0) && (Param[0]->Val->Integer < ACTUATORSETTINGS_CHANNELUPDATEFREQ_NUMELEM)) {
		ActuatorSettingsData data;
		ActuatorSettingsChannelUpdateFreqGet(data.ChannelUpdateFreq);
		data.ChannelUpdateFreq[Param[0]->Val->Integer] = Param[1]->Val->UnsignedInteger;
		ActuatorSettingsChannelUpdateFreqSet(data.ChannelUpdateFreq);
	}
}

/* PWMOutSet(int,int): set output pulse width of an unused output channel */
void SystemPWMOutSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	int16_t channel = Param[0]->Val->Integer;
	int16_t value = Param[1]->Val->Integer;

	// check channel range
	if ((channel < 0) || (channel >= ACTUATORSETTINGS_CHANNELNEUTRAL_NUMELEM))
		return;

	// check mixer settings
	MixerSettingsData mixerSettings;
	MixerSettingsGet(&mixerSettings);

	// this structure is equivalent to the UAVObjects for one mixer.
	typedef struct {
		uint8_t type;
		int8_t matrix[5];
	} __attribute__((packed)) Mixer_t;

	// base pointer to mixer vectors and types
	Mixer_t * mixers = (Mixer_t *)&mixerSettings.Mixer1Type;

	// the mixer has to be disabled for this channel.
	if (mixers[channel].type != MIXERSETTINGS_MIXER1TYPE_DISABLED)
		return;

	// check actuator settings
	ActuatorSettingsData actuatorSettings;
	ActuatorSettingsGet(&actuatorSettings);

	// the channel type has to be a PWM output.
	if (actuatorSettings.ChannelType[channel] != ACTUATORSETTINGS_CHANNELTYPE_PWM)
		return;

	actuatorSettings.ChannelMin[channel] = value;
	actuatorSettings.ChannelMax[channel] = value;
	actuatorSettings.ChannelNeutral[channel] = value;

	ActuatorSettingsSet(&actuatorSettings);
}

/* SystemPWMInGet(int): get the measured pulse width value of a PWM input */
#ifdef PIOS_INCLUDE_PWM
void SystemPWMInGet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	extern uintptr_t pios_rcvr_group_map[];
	ReturnValue->Val->Integer = PIOS_RCVR_Read(pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PWM], Param[0]->Val->Integer);
}
#endif

/* TxChannelValGet(int): get a tx value of the selected channel */
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
	{ SystemAccessLevelSet,	"void AccessLevelSet(int);" },
#ifdef PIOS_COM_PICOC
	{ SystemChangeBaud,		"void ChangeBaud(long);" },
#endif
	{ SystemTestValGet,		"int TestValGet();" },
	{ SystemTestValSet,		"void TestValSet(int);" },
#ifndef NO_FP
	{ SystemAccessoryValGet,"float AccessoryValGet(int);" },
	{ SystemAccessoryValSet,"void AccessoryValSet(int,float);" },
#endif
	{ SystemPWMFreqSet,		"void PWMFreqSet(int,unsigned int);" },
	{ SystemPWMOutSet,		"void PWMOutSet(int,int);" },
#ifdef PIOS_INCLUDE_PWM
	{ SystemPWMInGet,		"int PWMInGet(int);" },
#endif
	{ SystemTxChannelValGet,"int TxChannelValGet(int);" },
	{ NULL, NULL }
};


/**
 * accels.h
 */
#include "accels.h"

/* library functions */
#ifndef NO_FP
void Accels_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	AccelsData data;
	AccelsGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		double x;
		double y;
		double z;
		double temperature;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->x = data.x;
	pdata->y = data.y;
	pdata->z = data.z;
	pdata->temperature = data.temperature;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_accels[] =
{
#ifndef NO_FP
	{ Accels_Get,	"void AccelsGet(AccelsData *);" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_accels(Picoc *pc)
{
#ifndef NO_FP
	const char *definition = "typedef struct {"
		"float x;"
		"float y;"
		"float z;"
		"float temperature;"
	"} AccelsData;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);
#endif

	if (AccelsHandle() == NULL)
		ProgramFailNoParser(pc, "no accels");
}


/**
 * attitudeactual.h
 */
#include "attitudeactual.h"

/* library functions */
#ifndef NO_FP
void AttitudeActual_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	AttitudeActualData data;
	AttitudeActualGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		double Roll;
		double Pitch;
		double Yaw;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->Roll = data.Roll;
	pdata->Pitch = data.Pitch;
	pdata->Yaw = data.Yaw;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_attitudeactual[] =
{
#ifndef NO_FP
	{ AttitudeActual_Get,	"void AttitudeActualGet(AttitudeActualData *);" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_attitudeactual(Picoc *pc)
{
#ifndef NO_FP
	const char *definition = "typedef struct {"
		"float Roll;"
		"float Pitch;"
		"float Yaw;"
	"} AttitudeActualData;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);
#endif

	if (AttitudeActualHandle() == NULL)
		ProgramFailNoParser(pc, "no attitudeactual");
}


/**
 * baroaltitude.h
 */
#include "baroaltitude.h"

/* library functions */
#ifndef NO_FP
void BaroAltitude_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	BaroAltitudeData data;
	BaroAltitudeGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		double Altitude;
		double Temperature;
		double Pressure;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->Altitude = data.Altitude;
	pdata->Temperature = data.Temperature;
	pdata->Pressure = data.Pressure;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_baroaltitude[] =
{
#ifndef NO_FP
	{ BaroAltitude_Get,	"void BaroAltitudeGet(BaroAltitudeData *);" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_baroaltitude(Picoc *pc)
{
#ifndef NO_FP
	const char *definition = "typedef struct {"
		"float Altitude;"
		"float Temperature;"
		"float Pressure;"
	"} BaroAltitudeData;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);
#endif

	if (BaroAltitudeHandle() == NULL)
		ProgramFailNoParser(pc, "no baroaltitude");
}


/**
 * flightbatterystate.h
 */
#include "flightbatterystate.h"

/* library functions */
#ifndef NO_FP
void FlightBatteryState_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	FlightBatteryStateData data;
	FlightBatteryStateGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		double Voltage;
		double Current;
		double BoardSupplyVoltage;
		double PeakCurrent;
		double AvgCurrent;
		double ConsumedEnergy;
		double EstimatedFlightTime;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->Voltage = data.Voltage;
	pdata->Current = data.Current;
	pdata->BoardSupplyVoltage = data.BoardSupplyVoltage;
	pdata->PeakCurrent = data.PeakCurrent;
	pdata->AvgCurrent = data.AvgCurrent;
	pdata->ConsumedEnergy = data.ConsumedEnergy;
	pdata->EstimatedFlightTime = data.EstimatedFlightTime;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_flightbatterystate[] =
{
#ifndef NO_FP
	{ FlightBatteryState_Get,	"void FlightBatteryStateGet(FlightBatteryStateData *);" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_flightbatterystate(Picoc *pc)
{
#ifndef NO_FP
	const char *definition = "typedef struct {"
		"float Voltage;"
		"float Current;"
		"float BoardSupplyVoltage;"
		"float PeakCurrent;"
		"float AvgCurrent;"
		"float ConsumedEnergy;"
		"float EstimatedFlightTime;"
	"} FlightBatteryStateData;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);
#endif

	if (FlightBatteryStateHandle() == NULL)
		ProgramFailNoParser(pc, "no flightbatterystate");
}


/**
 * flightstatus.h
 */
#include "flightstatus.h"

/* library functions */
void FlightStatus_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	FlightStatusData data;
	FlightStatusGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		unsigned char Armed;
		unsigned char FlightMode;
		unsigned char ControlSource;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->Armed = data.Armed;
	pdata->FlightMode = data.FlightMode;
	pdata->ControlSource = data.ControlSource;
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_flightstatus[] =
{
	{ FlightStatus_Get,	"void FlightStatusGet(FlightBatteryStateData *);" },
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_flightstatus(Picoc *pc)
{
	const char *definition = "typedef struct {"
		"unsigned char Armed;"
		"unsigned char FlightMode;"
		"unsigned char ControlSource;"
	"} FlightStatusData;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);
}

/**
 * gpsposition.h
 */
#include "gpsposition.h"

/* library functions */
void GPSPosition_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
	return;

	GPSPositionData data;
	GPSPositionGet(&data);

	// use the same struct like picoc. see below
#ifndef NO_FP
	struct mystruct {
		long Latitude;
		long Longitude;
		double Altitude;
		double GeoidSeparation;
		double Heading;
		double Groundspeed;
		double PDOP;
		double HDOP;
		double VDOP;
		unsigned char Status;
		char Satellites;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->Latitude = data.Latitude;
	pdata->Longitude = data.Longitude;
	pdata->Altitude = data.Altitude;
	pdata->GeoidSeparation = data.GeoidSeparation;
	pdata->Heading = data.Heading;
	pdata->Groundspeed = data.Groundspeed;
	pdata->PDOP = data.PDOP;
	pdata->HDOP = data.HDOP;
	pdata->VDOP = data.VDOP;
	pdata->Status = data.Status;
	pdata->Satellites = data.Satellites;
#else
	struct mystruct {
		long Latitude;
		long Longitude;
		unsigned char Status;
		char Satellites;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->Latitude = data.Latitude;
	pdata->Longitude = data.Longitude;
	pdata->Status = data.Status;
	pdata->Satellites = data.Satellites;
#endif
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_gpsposition[] =
{
	{ GPSPosition_Get,	"void GPSPositionGet(GPSPositionData *);" },
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_gpsposition(Picoc *pc)
{
#ifndef NO_FP
	const char *definition = "typedef struct {"
		"long Latitude;"
		"long Longitude;"
		"float Altitude;"
		"float GeoidSeparation;"
		"float Heading;"
		"float GroundSpeed;"
		"float PDOP;"
		"float HDOP;"
		"float VDOP;"
		"unsigned char Status;"
		"char Satellites;"
	"} GPSPositionData;";
#else
	const char *definition = "typedef struct {"
		"long Latitude;"
		"long Longitude;"
		"unsigned char Status;"
		"char Satellites;"
	"}GPSPositionData;";
#endif
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);

	if (GPSPositionHandle() == NULL)
		ProgramFailNoParser(pc, "no gpsposition");
}


/**
 * gyros.h
 */
#include "gyros.h"

/* library functions */
#ifndef NO_FP
void Gyros_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
	return;

	GyrosData data;
	GyrosGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		double x;
		double y;
		double z;
		double temperature;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->x = data.x;
	pdata->y = data.y;
	pdata->z = data.z;
	pdata->temperature = data.temperature;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_gyros[] =
{
#ifndef NO_FP
	{ Gyros_Get,	"void GyrosGet(GyrosData *);" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_gyros(Picoc *pc)
{
	const char *definition = "typedef struct {"
		"float x;"
		"float y;"
		"float z;"
		"float temperature;"
	"} GyrosData;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);

	if (GyrosHandle() == NULL)
		ProgramFailNoParser(pc, "no gyros");
}


/**
 * magnetometer.h
 */
#include "magnetometer.h"

/* library functions */
#ifndef NO_FP
void Magnetometer_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	MagnetometerData data;
	MagnetometerGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		double x;
		double y;
		double z;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->x = data.x;
	pdata->y = data.y;
	pdata->z = data.z;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_magnetometer[] =
{
#ifndef NO_FP
	{ Magnetometer_Get,	"void MagnetometerGet(MagnetometerData *);" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_magnetometer(Picoc *pc)
{
	const char *definition = "typedef struct {"
		"float x;"
		"float y;"
		"float z;"
	"} MagnetometerData;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);

	if (MagnetometerHandle() == NULL)
		ProgramFailNoParser(pc, "no magnetometer");
}


/**
 * manualcontrol.h
 */
#include "manualcontrolsettings.h"

/* library functions */
void FlightModePositionSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (!security(1))
		return;
	if ((Param[0]->Val->Integer >= 0) && (Param[0]->Val->Integer < MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_NUMELEM)) {
		ManualControlSettingsData data;
		ManualControlSettingsFlightModePositionGet(data.FlightModePosition);
		data.FlightModePosition[Param[0]->Val->Integer] = Param[1]->Val->Integer;
		ManualControlSettingsFlightModePositionSet(data.FlightModePosition);
	}
}

void Stabilization1SettingsSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (!security(1))
		return;
	ManualControlSettingsData data;
	ManualControlSettingsStabilization1SettingsGet(data.Stabilization1Settings);
	data.Stabilization1Settings[MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_ROLL] = Param[0]->Val->Integer;
	data.Stabilization1Settings[MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_PITCH] = Param[1]->Val->Integer;
	data.Stabilization1Settings[MANUALCONTROLSETTINGS_STABILIZATION1SETTINGS_YAW] = Param[2]->Val->Integer;
	ManualControlSettingsStabilization1SettingsSet(data.Stabilization1Settings);
}

void Stabilization2SettingsSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (!security(1))
		return;
	ManualControlSettingsData data;
	ManualControlSettingsStabilization2SettingsGet(data.Stabilization2Settings);
	data.Stabilization2Settings[MANUALCONTROLSETTINGS_STABILIZATION2SETTINGS_ROLL] = Param[0]->Val->Integer;
	data.Stabilization2Settings[MANUALCONTROLSETTINGS_STABILIZATION2SETTINGS_PITCH] = Param[1]->Val->Integer;
	data.Stabilization2Settings[MANUALCONTROLSETTINGS_STABILIZATION2SETTINGS_YAW] = Param[2]->Val->Integer;
	ManualControlSettingsStabilization2SettingsSet(data.Stabilization2Settings);
}

void Stabilization3SettingsSet(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (!security(1))
		return;
	ManualControlSettingsData data;
	ManualControlSettingsStabilization3SettingsGet(data.Stabilization3Settings);
	data.Stabilization3Settings[MANUALCONTROLSETTINGS_STABILIZATION3SETTINGS_ROLL] = Param[0]->Val->Integer;
	data.Stabilization3Settings[MANUALCONTROLSETTINGS_STABILIZATION3SETTINGS_PITCH] = Param[1]->Val->Integer;
	data.Stabilization3Settings[MANUALCONTROLSETTINGS_STABILIZATION3SETTINGS_YAW] = Param[2]->Val->Integer;
	ManualControlSettingsStabilization3SettingsSet(data.Stabilization3Settings);
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_manualcontrol[] =
{
	{ FlightModePositionSet,		"void FlightModeSet(int,int);" },
	{ Stabilization1SettingsSet,	"void Stabilized1Set(int,int,int);" },
	{ Stabilization2SettingsSet,	"void Stabilized2Set(int,int,int);" },
	{ Stabilization3SettingsSet,	"void Stabilized3Set(int,int,int);" },
	{ NULL, NULL }
};


/**
 * positionactual.h
 */
#include "positionactual.h"

/* library functions */
#ifndef NO_FP
void PositionActual_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	PositionActualData data;
	PositionActualGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		double North;
		double East;
		double Down;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->North = data.North;
	pdata->East = data.East;
	pdata->Down = data.Down;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_positionactual[] =
{
#ifndef NO_FP
	{ PositionActual_Get,	"void PositionActualGet(PositionActualData *);" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_positionactual(Picoc *pc)
{
	const char *definition = "typedef struct {"
		"float North;"
		"float East;"
		"float Down;"
	"} PositionActualData;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);

	if (PositionActualHandle() == NULL)
		ProgramFailNoParser(pc, "no positionactual");
}


/* list all includes */
void PlatformLibraryInit(Picoc *pc)
{
	// ensure we run in user state at startup
	accesslevel = 0;

#ifndef NO_STRING_FUNCTIONS
	IncludeRegister(pc, "string.h", NULL, &PlatformLibrary_string[0], NULL);
#endif
#ifndef NO_FP
	IncludeRegister(pc, "math.h", &PlatformLibrarySetup_math, &PlatformLibrary_math[0], NULL);
#endif
	IncludeRegister(pc, "system.h", NULL, &PlatformLibrary_system[0], NULL);
	IncludeRegister(pc, "accels.h", &PlatformLibrarySetup_accels, &PlatformLibrary_accels[0], NULL);
	IncludeRegister(pc, "attitudeactual.h", &PlatformLibrarySetup_attitudeactual, &PlatformLibrary_attitudeactual[0], NULL);
	IncludeRegister(pc, "baroaltitude.h", &PlatformLibrarySetup_baroaltitude, &PlatformLibrary_baroaltitude[0], NULL);
	IncludeRegister(pc, "flightbatterystate.h", &PlatformLibrarySetup_flightbatterystate, &PlatformLibrary_flightbatterystate[0], NULL);
	IncludeRegister(pc, "flightstatus.h", &PlatformLibrarySetup_flightstatus, &PlatformLibrary_flightstatus[0], NULL);
	IncludeRegister(pc, "gpsposition.h", &PlatformLibrarySetup_gpsposition, &PlatformLibrary_gpsposition[0], NULL);
	IncludeRegister(pc, "gyros.h", &PlatformLibrarySetup_gyros, &PlatformLibrary_gyros[0], NULL);
	IncludeRegister(pc, "magnetometer.h", &PlatformLibrarySetup_magnetometer, &PlatformLibrary_magnetometer[0], NULL);
	IncludeRegister(pc, "manualcontrol.h", NULL, &PlatformLibrary_manualcontrol[0], NULL);
	IncludeRegister(pc, "positionactual.h", &PlatformLibrarySetup_positionactual, &PlatformLibrary_positionactual[0], NULL);
}

#endif /* PIOS_INCLUDE_PICOC */

/**
 * @}
 * @}
 */
