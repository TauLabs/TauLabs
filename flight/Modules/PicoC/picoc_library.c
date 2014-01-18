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
#if defined(PIOS_INCLUDE_PICOC)

#include "openpilot.h"
#include "picoc_port.h"
#include "picocstatus.h"
#include "flightstatus.h"

// Private constants
#define UAVO_GET 1
#define UAVO_SET 2

// Private variables
static int uavoGetValue = UAVO_GET;
static int uavoSetValue = UAVO_SET;

/**
 * PicoC platform depending library functions for UAVOs
 * normaly stored in library_xxx.c
 */

/**
 * picoc library functions
 * available after #include "picoc.h"
 */

/* void delay(int): sleep for given ms-value */
 void Cdelay(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs) 
{
	int16_t value = Param[0]->Val->Integer;

	if (value > 0) {
		vTaskDelay(MS2TICKS(value));
	}
}

/* void sync(int): synchronize an interval by given ms-value */
 void Csync(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs) 
{
	static portTickType lastSysTime;
	int16_t value = Param[0]->Val->Integer;

	if ((lastSysTime == 0) || (value == 0)) {
		lastSysTime = xTaskGetTickCount();
	}
	if (value > 0) {
		vTaskDelayUntil(&lastSysTime, MS2TICKS(value));
	}
}

/* int armed: returns armed status */
 void Carmed(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs) 
{
	FlightStatusData uavo;
	FlightStatusArmedGet(&uavo.Armed);
	ReturnValue->Val->Integer = (uavo.Armed == FLIGHTSTATUS_ARMED_ARMED);
}

/* void changebaud(long): changes the speed of picoc serial port */
void Cchangebaud(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs) 
{
#ifdef PIOS_COM_PICOC
	uint32_t value = Param[0]->Val->LongInteger;

	if ((PIOS_COM_PICOC) && (value >0) && (value <=115200)) {
		PIOS_COM_ChangeBaud(PIOS_COM_PICOC, value);
	}
#endif
}

/* int picoctest(int,int): set and get picoc test value. prototype function for other UAVOs */
void Ctestvalue(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs) 
{
	PicoCStatusData uavo;
	switch (Param[0]->Val->Integer) {
	case UAVO_GET:
		PicoCStatusTestValueGet(&uavo.TestValue);
		ReturnValue->Val->Integer = uavo.TestValue;
		break;
	case UAVO_SET:
		uavo.TestValue = Param[1]->Val->Integer;
		PicoCStatusTestValueSet(&uavo.TestValue);
		break;
	default:
		;
	}
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary_picoc[] =
{
	{ Cdelay,		"void delay(int);"},
	{ Csync,		"void sync(int);"},
	{ Carmed,		"int armed(void);"},
	{ Cchangebaud,	"void changebaud(long);"},
	{ Ctestvalue,	"int testvalue(int,int);"},
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup_picoc(Picoc *pc)
{
	// define some handy values for function handling
	VariableDefinePlatformVar(pc, NULL, "GET", &pc->IntType, (union AnyValue *)&uavoGetValue, FALSE);
	VariableDefinePlatformVar(pc, NULL, "SET", &pc->IntType, (union AnyValue *)&uavoSetValue, FALSE);
}

void PlatformLibraryInit(Picoc *pc)
{
    IncludeRegister(pc, "picoc.h", &PlatformLibrarySetup_picoc, &PlatformLibrary_picoc[0], NULL);
}


#endif /* PIOS_INCLUDE_PICOC */


/**
 * @}
 * @}
 */
