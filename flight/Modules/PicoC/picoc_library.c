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
#include "pios_thread.h"

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
		PIOS_Thread_Sleep(Param[0]->Val->Integer);
	}
}

/* void sync(int): synchronize an interval by given ms-value */
void SystemSync(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	static uint32_t lastSysTime;
	if ((lastSysTime == 0) || (Param[0]->Val->Integer == 0)) {
		lastSysTime = PIOS_Thread_Systime();
	}
	if (Param[0]->Val->Integer > 0) {
		PIOS_Thread_Sleep_Until(&lastSysTime, Param[0]->Val->Integer);
	}
}

/* unsigned long time(): returns actual systemtime as ms-value */
void SystemTime(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->UnsignedLongInteger = PIOS_Thread_Systime();
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
	if ((PIOS_COM_PICOC) && (Param[0]->Val->UnsignedLongInteger > 0)) {
		PIOS_COM_ChangeBaud(PIOS_COM_PICOC, Param[0]->Val->UnsignedLongInteger);
	}
}
/* long SendBuffer(unsigned char *,unsigned int): sends a buffer content to picoc serial port */
void SystemSendBuffer(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if ((PIOS_COM_PICOC) && (Param[0]->Val->Pointer != NULL)) {
		uint8_t *buffer = Param[0]->Val->Pointer;
		uint16_t buf_len = Param[1]->Val->UnsignedInteger;
		ReturnValue->Val->LongInteger = 0;
		while (buf_len > 0) {
			int32_t rc = PIOS_COM_SendBufferNonBlocking(PIOS_COM_PICOC, buffer, buf_len);
			if (rc > 0) {
				buf_len -= rc;
				buffer += rc;
				ReturnValue->Val->LongInteger += rc;
			} else if (rc == 0) {
				PIOS_Thread_Sleep(1);
			} else {
				ReturnValue->Val->LongInteger = rc;
				return;
			}
		}
	} else {
		ReturnValue->Val->LongInteger = -1;
	}
}
/* long ReceiveBuffer(unsigned char *,unsigned int,unsigned long): receives buffer from picoc serial port */
void SystemReceiveBuffer(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if ((PIOS_COM_PICOC) && (Param[0]->Val->Pointer != NULL)) {
		uint8_t *buffer = Param[0]->Val->Pointer;
		uint16_t buf_len = Param[1]->Val->UnsignedInteger;
		uint32_t timeout = Param[2]->Val->UnsignedLongInteger;
		ReturnValue->Val->LongInteger = 0;
		while (buf_len > 0) {
			uint16_t rc = PIOS_COM_ReceiveBuffer(PIOS_COM_PICOC, buffer, buf_len, 0);
			if (rc > 0) {
				buf_len -= rc;
				buffer += rc;
				ReturnValue->Val->LongInteger += rc;
			} else if (timeout-- > 0) {
				PIOS_Thread_Sleep(1);
			} else {
				return;
			}
		}
	} else {
		ReturnValue->Val->LongInteger = -1;
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
	if ((Param[0]->Val->Integer >= 0) && (Param[0]->Val->Integer < ACTUATORSETTINGS_TIMERUPDATEFREQ_NUMELEM)) {
		ActuatorSettingsData data;
		ActuatorSettingsTimerUpdateFreqGet(data.TimerUpdateFreq);
		data.TimerUpdateFreq[Param[0]->Val->Integer] = Param[1]->Val->UnsignedInteger;
		ActuatorSettingsTimerUpdateFreqSet(data.TimerUpdateFreq);
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

#ifdef PIOS_INCLUDE_I2C
/* void int I2CRead(unsigned char,unsigned char, void *, unsigned int): read bytes from i2c slave */
void SystemI2CRead(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	uint32_t i2c_adapter;
	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = Param[1]->Val->UnsignedCharacter,
			.rw   = PIOS_I2C_TXN_READ,
			.len  = Param[3]->Val->UnsignedInteger,
			.buf  = Param[2]->Val->Pointer,
		},
	};

	switch(Param[0]->Val->Integer) {
#if defined(PIOS_I2C_ADAPTER_0)
		case 0:
			i2c_adapter = PIOS_I2C_ADAPTER_0;
			break;
#endif
#if defined(PIOS_I2C_ADAPTER_1)
		case 1:
			i2c_adapter = PIOS_I2C_ADAPTER_1;
			break;
#endif
#if defined(PIOS_I2C_ADAPTER_2)
		case 2:
			i2c_adapter = PIOS_I2C_ADAPTER_2;
			break;
#endif
		default:
			i2c_adapter = 0;
	}

	if ((i2c_adapter) && (Param[2]->Val->Pointer != NULL)) {
		ReturnValue->Val->Integer = PIOS_I2C_Transfer(i2c_adapter, txn_list, NELEMENTS(txn_list));
	}
	else {
		ReturnValue->Val->Integer = -1;
	}
}

/* void int I2CWrite(unsigned char,unsigned char, void *, unsigned int): write bytes to i2c slave */
void SystemI2CWrite(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	uint32_t i2c_adapter;
	const struct pios_i2c_txn txn_list[] = {
		{
			.info = __func__,
			.addr = Param[1]->Val->UnsignedCharacter,
			.rw   = PIOS_I2C_TXN_WRITE,
			.len  = Param[3]->Val->UnsignedInteger,
			.buf  = Param[2]->Val->Pointer,
		},
	};

	switch(Param[0]->Val->Integer) {
#if defined(PIOS_I2C_ADAPTER_0)
		case 0:
			i2c_adapter = PIOS_I2C_ADAPTER_0;
			break;
#endif
#if defined(PIOS_I2C_ADAPTER_1)
		case 1:
			i2c_adapter = PIOS_I2C_ADAPTER_1;
			break;
#endif
#if defined(PIOS_I2C_ADAPTER_2)
		case 2:
			i2c_adapter = PIOS_I2C_ADAPTER_2;
			break;
#endif
		default:
			i2c_adapter = 0;
	}

	if ((i2c_adapter) && (Param[2]->Val->Pointer != NULL)) {
		ReturnValue->Val->Integer = PIOS_I2C_Transfer(i2c_adapter, txn_list, NELEMENTS(txn_list));
	}
	else {
		ReturnValue->Val->Integer = -1;
	}
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_system[] =
{
	{ SystemDelay,			"void delay(int);" },
	{ SystemSync,			"void sync(int);" },
	{ SystemTime,			"unsigned long time();" },
	{ SystemArmed,			"int armed();" },
	{ SystemAccessLevelSet,	"void AccessLevelSet(int);" },
#ifdef PIOS_COM_PICOC
	{ SystemChangeBaud,		"void ChangeBaud(unsigned long);" },
	{ SystemSendBuffer,		"long SendBuffer(void *,unsigned int);" },
	{ SystemReceiveBuffer,	"long ReceiveBuffer(void *,unsigned int,unsigned long);" },
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
#ifdef PIOS_INCLUDE_I2C
	{ SystemI2CRead,		"int i2c_read(unsigned char,unsigned char, void *,unsigned int);" },
	{ SystemI2CWrite,		"int i2c_write(unsigned char,unsigned char, void *,unsigned int);" },
#endif
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
	{ FlightStatus_Get,	"void FlightStatusGet(FlightStatusData *);" },
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
	"} GPSPositionData;";
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
 * pathdesired.h
 */
#include "pathdesired.h"

/* library functions */
#ifndef NO_FP
void PathDesired_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	PathDesiredData data;
	PathDesiredGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		double Start[3];
		double End[3];
		double StartingVelocity;
		double EndingVelocity;
		double ModeParameters;
		unsigned char Mode;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->Start[0] = data.Start[0];
	pdata->Start[1] = data.Start[1];
	pdata->Start[2] = data.Start[2];
	pdata->End[0] = data.End[0];
	pdata->End[1] = data.End[1];
	pdata->End[2] = data.End[2];
	pdata->StartingVelocity = data.StartingVelocity;
	pdata->EndingVelocity = data.EndingVelocity;
	pdata->ModeParameters = data.ModeParameters;
	pdata->Mode = data.Mode;
}

void PathDesired_Set(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	PathDesiredData data;
	PathDesiredGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		double Start[3];
		double End[3];
		double StartingVelocity;
		double EndingVelocity;
		double ModeParameters;
		unsigned char Mode;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	data.Start[0] = pdata->Start[0];
	data.Start[1] = pdata->Start[1];
	data.Start[2] = pdata->Start[2];
	data.End[0] = pdata->End[0];
	data.End[1] = pdata->End[1];
	data.End[2] = pdata->End[2];
	data.StartingVelocity = pdata->StartingVelocity;
	data.EndingVelocity = pdata->EndingVelocity;
	data.ModeParameters = pdata->ModeParameters;
	data.Mode = pdata->Mode;

	PathDesiredSet(&data);
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_pathdesired[] =
{
#ifndef NO_FP
	{ PathDesired_Get,	"void PathDesiredGet(PathDesiredData *);" },
	{ PathDesired_Set,	"void PathDesiredSet(PathDesiredData *);" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_pathdesired(Picoc *pc)
{
	const char *definition = "typedef struct {"
		"float Start[3];"
		"float End[3];"
		"float StartingVelocity;"
		"float EndingVelocity;"
		"float ModeParameters;"
		"unsigned char Mode;"
	"} PathDesiredData;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);

	if (PathDesiredHandle() == NULL)
		ProgramFailNoParser(pc, "no pathdesired");
}


/**
 * pathstatus.h
 */
#include "pathstatus.h"

/* library functions */
#ifndef NO_FP
void PathStatus_Get(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	PathStatusData data;
	PathStatusGet(&data);

	// use the same struct like picoc. see below
	struct mystruct {
		double fractional_progress;
		double error;
		unsigned char Status;
	} *pdata;
	pdata = Param[0]->Val->Pointer;
	pdata->fractional_progress = data.fractional_progress;
	pdata->error = data.error;
	pdata->Status = data.Status;
}
#endif

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_pathstatus[] =
{
#ifndef NO_FP
	{ PathStatus_Get,	"void PathStatusGet(PathStatusData *);" },
#endif
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_pathstatus(Picoc *pc)
{
	const char *definition = "typedef struct {"
		"float fractional_progress;"
		"float error;"
		"unsigned char Status;"
	"} PathStatusData;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);

	if (PathStatusHandle() == NULL)
		ProgramFailNoParser(pc, "no pathstatus");
}


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


/**
 * pid.h
 */
#ifndef NO_FP
/* less comments here. all functions are functional identical to math/pid.c */

/* library definitions */
#define bound_min_max(val,min,max) ((val < min) ? min : (val > max) ? max : val)
#define bound_sym(val,range) ((val < -range) ? -range : (val > range) ? range : val)

struct pid {
	double p;
	double i;
	double d;
	double iLim;
	double iAccumulator;
	double lastErr;
	double lastDer;
	double dTau;
};

/* library functions */
void PID_apply(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	struct pid *pid = Param[0]->Val->Pointer;
	double err = Param[1]->Val->FP;
	double dT = Param[2]->Val->FP;

	if (pid->i != 0) {
		pid->iAccumulator += err * (pid->i * dT);
		pid->iAccumulator = bound_sym(pid->iAccumulator, pid->iLim);
	}

	double diff = (err - pid->lastErr);
	double dterm = 0;
	pid->lastErr = err;
	if(pid->d && dT)
	{
		dterm = pid->lastDer +  dT / ( dT + pid->dTau) * ((diff * pid->d / dT) - pid->lastDer);
		pid->lastDer = dterm;
	}
 
	ReturnValue->Val->FP = (err * pid->p) + pid->iAccumulator + dterm;
}

void PID_apply_antiwindup(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	struct pid *pid = Param[0]->Val->Pointer;
	double err = Param[1]->Val->FP;
	double min_bound = Param[2]->Val->FP;
	double max_bound = Param[3]->Val->FP;
	double dT = Param[4]->Val->FP;

	if (pid->i != 0) {
		pid->iAccumulator += err * (pid->i * dT);
	}

	double diff = (err - pid->lastErr);
	double dterm = 0;
	pid->lastErr = err;
	if(pid->d && dT)
	{
		dterm = pid->lastDer +  dT / ( dT + pid->dTau) * ((diff * pid->d / dT) - pid->lastDer);
		pid->lastDer = dterm;
	}

	double ideal_output = ((err * pid->p) + pid->iAccumulator + dterm);
	double saturation = 0;
	if (ideal_output > max_bound) {
		saturation = max_bound - ideal_output;
		ideal_output = max_bound;
	} else if (ideal_output < min_bound) {
		saturation = min_bound - ideal_output;
		ideal_output = min_bound;
	}
	// Use Kt 10x Ki
	pid->iAccumulator += saturation * (pid->i * 10.0 * dT);
	pid->iAccumulator = bound_sym(pid->iAccumulator, pid->iLim);

	ReturnValue->Val->FP = ideal_output;
}

void PID_apply_setpoint(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	struct pid *pid = Param[0]->Val->Pointer;
	double setpoint = Param[1]->Val->FP;
	double measured = Param[2]->Val->FP;
	double dT = Param[3]->Val->FP;

	double err = setpoint - measured;

	if (pid->i != 0) {
		pid->iAccumulator += err * (pid->i * dT);
		pid->iAccumulator = bound_sym(pid->iAccumulator, pid->iLim);
	}

	double dterm = 0;
	double diff = setpoint - measured - pid->lastErr;
	pid->lastErr = setpoint - measured;
	if(pid->d && dT)
	{
		dterm = pid->lastDer +  dT / ( dT + pid->dTau) * ((diff * pid->d / dT) - pid->lastDer);
		pid->lastDer = dterm;
	}

	ReturnValue->Val->FP = (err * pid->p) + pid->iAccumulator + dterm;
}

void PID_zero(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	struct pid *pid = Param[0]->Val->Pointer;

	pid->iAccumulator = 0;
	pid->lastErr = 0;
	pid->lastDer = 0;
}

void PID_configure(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	if (Param[0]->Val->Pointer == NULL)
		return;

	struct pid *pid = Param[0]->Val->Pointer;

	pid->p = Param[1]->Val->FP;
	pid->i = Param[2]->Val->FP;
	pid->d = Param[3]->Val->FP;
	pid->iLim = Param[4]->Val->FP;
	double cutoff = bound_min_max(Param[5]->Val->FP,1,100);
	pid->dTau = 1.0 / (2 * 3.14159265358979323846 * cutoff);
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_pid[] =
{
	{ PID_apply,			"float pid_apply(pid *,float,float);" },
	{ PID_apply_antiwindup,	"float pid_apply_antiwindup(pid *,float,float,float,float);" },
	{ PID_apply_setpoint,	"float pid_apply_setpoint(pid *,float,float,float);" },
	{ PID_zero,				"void pid_zero(pid *);" },
	{ PID_configure,		"void pid_configure(pid *,float,float,float,float,float);" },
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_pid(Picoc *pc)
{
	const char *definition = "typedef struct {"
		"float p;"
		"float i;"
		"float d;"
		"float iLim;"
		"float iAccumulator;"
		"float lastErr;"
		"float lastDer;"
		"float dTau;"
	"} pid;";
	PicocParse(pc, "mylib", definition, strlen(definition), TRUE, TRUE, FALSE, FALSE);
}
#endif


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
	IncludeRegister(pc, "pathdesired.h", &PlatformLibrarySetup_pathdesired, &PlatformLibrary_pathdesired[0], NULL);
	IncludeRegister(pc, "pathstatus.h", &PlatformLibrarySetup_pathstatus, &PlatformLibrary_pathstatus[0], NULL);
	IncludeRegister(pc, "positionactual.h", &PlatformLibrarySetup_positionactual, &PlatformLibrary_positionactual[0], NULL);
#ifndef NO_FP
	IncludeRegister(pc, "pid.h", &PlatformLibrarySetup_pid, &PlatformLibrary_pid[0], NULL);
#endif
}

#endif /* PIOS_INCLUDE_PICOC */

/**
 * @}
 * @}
 */
