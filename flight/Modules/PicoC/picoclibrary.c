/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup PicoC Interpreter Module
 * @{ 
 *
 * @file       picoclibrary.c
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
#include "picocsettings.h"
#include "picocstate.h"
#include "flightstatus.h"
#include "picoc.h"

// Private constants
#define UAVO_READ 0
#define UAVO_WRITE 1

// Private variables
static int uavoReadValue = UAVO_READ;
static int uavoWriteValue = UAVO_WRITE;

/**
 * PicoC platform depending library functions for UAVOs
 * normaly stored in library_xxx.c
 */

/* *
 * picoctest: for simple program debug of store a value
 * prototype for UAVO communication
 * first parameter is access mode
 * second parameter is the write value
 * function returns the old value of the UAVO
 */
void Cpicoctest(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs) 
{
	// Define value variable for UAVO communication. The type depends on the UAVO.
	int16_t value;

	// Always get the actual value as return value.
	PicoCStateTestValueGet(&value);
	ReturnValue->Val->Integer = value;

	// Write the new value, if wanted.
	if (Param[0]->Val->Integer == UAVO_WRITE) {
		value = Param[1]->Val->Integer;
		PicoCStateTestValueSet(&value);
	}
}

/* delay(int): sleep for given ms-value */
 void Cdelay(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs) 
{
	vTaskDelay(Param[0]->Val->Integer);
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary[] =
{
	{ Cpicoctest,	"int picoctest(int,int);"},
	{ Cdelay,		"void delay(int);"},
	{ NULL, NULL }
};

/* this is called when the header file is included */
void PlatformLibrarySetup(Picoc *pc)
{
	// define some values for uavo handling
	VariableDefinePlatformVar(pc, NULL, "URD", &pc->IntType, (union AnyValue *)&uavoReadValue, FALSE);
	VariableDefinePlatformVar(pc, NULL, "UWR", &pc->IntType, (union AnyValue *)&uavoWriteValue, FALSE);
}

void PlatformLibraryInit(Picoc *pc)
{
    IncludeRegister(pc, "uavo.h", &PlatformLibrarySetup, &PlatformLibrary[0], NULL);
}

#endif /* PIOS_INCLUDE_PICOC */

/**
 * @}
 * @}
 */
