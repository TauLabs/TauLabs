/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup PicoC Interpreter Module
 * @{ 
 *
 * @file       picocmodule.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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
#include "modulesettings.h"
#include <setjmp.h>

// Private constants
#define PICOC_HEAP_SIZE				(16*1024)
#define STACK_SIZE_BYTES			(PICOC_HEAP_SIZE + 16000)
#define TASK_PRIORITY				(tskIDLE_PRIORITY + 1)
#define TASK_RATE_HZ				1

// Private variables
static xTaskHandle picocTaskHandle;
static uintptr_t picocPort;
static bool module_enabled;
static jmp_buf picocExitBuf;

// Private functions
static void picocTask(void *parameters);
static void updateSpeedSettings();
int picoc(const char * SourceStr, int StackSize);
// void pprintf(const char *format, ...);

/**
 * start the module
 * \return -1 if start failed
 * \return 0 on success
 */
static int32_t picocStart(void)
{
	if (module_enabled) {
		// Start task
		xTaskCreate(picocTask, (signed char *) "PicoC",
				STACK_SIZE_BYTES / 4, NULL, TASK_PRIORITY,
				&picocTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_PICOC,
				picocTaskHandle);
		return 0;
	}
	return -1;
}

/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t picocInitialize(void)
{
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);

	picocPort = PIOS_COM_PICOC;

	if (module_state[MODULESETTINGS_ADMINSTATE_PICOC] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
		PicoCSettingsInitialize();
		PicoCStateInitialize();
	} else {
		module_enabled = false;
	}
	return 0;
}
MODULE_INITCALL( picocInitialize, picocStart)

/**
 * Main task. It does not return.
 */
static void picocTask(void *parameters) {

	const char *DemoC = "void test() {for (int i=0; i<10; i++) printf(\"i=%d\\n\", i); } test();";
	PicoCSettingsData picocsettings;
	FlightStatusData flightstatus;
	bool startup;
	int16_t retval = 0;

	updateSpeedSettings();
	while (1) {
		PicoCSettingsGet(&picocsettings);
		FlightStatusGet(&flightstatus);

		switch (picocsettings.Startup) {
		case PICOCSETTINGS_STARTUP_DISABLED:
			startup = false;
		case PICOCSETTINGS_STARTUP_ONBOOT:
			startup = true;
			break;
		case PICOCSETTINGS_STARTUP_WHENARMED:
			startup = (flightstatus.Armed == FLIGHTSTATUS_ARMED_ARMED);
			break;
		default:
			startup = false;
		}

		if (startup) {
			switch (picocsettings.Source) {
			case PICOCSETTINGS_SOURCE_DEMO:
				retval = picoc(DemoC, PICOC_HEAP_SIZE);
				break;
			case PICOCSETTINGS_SOURCE_INTERACTIVE:
				retval = picoc(NULL, PICOC_HEAP_SIZE);
				break;
			case PICOCSETTINGS_SOURCE_FILE:
				// no filesystem yet.
				retval = -3;
				break;
			default:
				retval = 0;
			}
			PicoCStateExitValueSet(&retval);
		}

		vTaskDelay(10);
	}
}

/**
 * update picoc module settings
 */
static void updateSpeedSettings()
{
	// if there is a com port, setup its speed.
	if (picocPort) {
		// Retrieve settings
		uint8_t speed;
		PicoCSettingsComSpeedGet(&speed);

		// Set port speed
		switch (speed) {
		case PICOCSETTINGS_COMSPEED_2400:
			PIOS_COM_ChangeBaud(picocPort, 2400);
			break;
		case PICOCSETTINGS_COMSPEED_4800:
			PIOS_COM_ChangeBaud(picocPort, 4800);
			break;
		case PICOCSETTINGS_COMSPEED_9600:
			PIOS_COM_ChangeBaud(picocPort, 9600);
			break;
		case PICOCSETTINGS_COMSPEED_19200:
			PIOS_COM_ChangeBaud(picocPort, 19200);
			break;
		case PICOCSETTINGS_COMSPEED_38400:
			PIOS_COM_ChangeBaud(picocPort, 38400);
			break;
		case PICOCSETTINGS_COMSPEED_57600:
			PIOS_COM_ChangeBaud(picocPort, 57600);
			break;
		case PICOCSETTINGS_COMSPEED_115200:
			PIOS_COM_ChangeBaud(picocPort, 115200);
			break;
		}
	}
}


/**
 * picoc implemetation
 * This is needed to compile picoc without a makefile.
 * all original picoc source files are in the include folder.
 */

// picoc related defines
#define HEAP_SIZE PICOC_HEAP_SIZE	/* space for the heap and the stack */
//#define USE_MALLOC_STACK	/* stack is allocated using malloc() */
//#define USE_MALLOC_HEAP		/* heap is allocated using malloc() */
#define BUILTIN_MINI_STDLIB
#define PICOC_LIBRARY
#define NO_CTYPE
#define NO_DEBUGGER
#define NO_MALLOC
#define NO_CALLOC
#define NO_REALLOC
#define NO_STRING_FUNCTIONS
#define malloc pvPortMalloc
#define calloc(a,b) pvPortMalloc(a*b)
#define free vPortFree
#define PicocPlatformSetExitPoint(pc) setjmp(picocExitBuf)
#define assert(x)

// include needed picoc sources
#include "picoc.h"
#include "picoc.c"
#include "table.c"
#include "lex.c"
#include "parse.c"
#include "expression.c"
#include "heap.c"
#include "type.c"
#include "variable.c"
#include "clibrary.c"
#include "platform.c"
#include "include.c"
#include "debug.c"


/**
 * picoc main program
 * parses sourcestring or switches to interactive mode
 * returns the exit() value
 */
 
int picoc(const char * SourceStr, int StackSize)
{
	Picoc pc;
	PicocInitialise(&pc, StackSize);

	if (PicocPlatformSetExitPoint(&pc)) {
		// we get here, if an error occures or 'exit();' was called.
		PicocCleanup(&pc);
		return pc.PicocExitValue;
	}

	if (SourceStr) {
		PicocParse(&pc, "nofile", SourceStr, strlen(SourceStr), TRUE, TRUE, FALSE, FALSE);
	} else {
		PicocParseInteractive(&pc);
	}

	PicocCleanup(&pc);
	return pc.PicocExitValue;
}


/**
 * basic functions for virtual stdIO
 * uses a optional USART or telemetry tunnel for communication
 */

/* get a character from stdIn or telemetry link */
uint8_t getch()
{
	uint8_t ch = 0;
	PicoCStateData state;

	while (1) {
		// try to get a char from USART
		if (picocPort) {
			if (PIOS_COM_ReceiveBuffer(picocPort, &ch, 1, 0) == 1) {
				break;
			}
		}
		// if there is a telemetry link, try it
		PicoCStateGet(&state);
		if ((state.LinkTimeout) && (state.GetChar)) {
			ch = state.GetChar;
			state.GetChar = 0;
			PicoCStateGetCharSet(&state.GetChar);
			break;
		}
		if (state.LinkTimeout) {
			state.LinkTimeout--;
			PicoCStateLinkTimeoutSet(&state.LinkTimeout);
		}
		if ((!picocPort) && (!state.LinkTimeout)) {
			// there is no USART and telemetry link is lost. end here.
			ch = 0;
			break;
		}
		if (state.LinkTimeout) {
			vTaskDelay(1000);
		}
		else {
			vTaskDelay(1);
		}
	}
	return ch;
}

/* put a character to stdOut */
void putch(uint8_t ch)
{
	PicoCStateData state;

	if (picocPort)
		PIOS_COM_SendBuffer(picocPort, &ch, sizeof(ch));

	// if there is a telemetrylink, send it to gcs too.
	while(1) {
		PicoCStateGet(&state);
		if ((state.LinkTimeout) && (!state.PutChar)) {
			PicoCStatePutCharSet(&ch);
			break;
		}
		if (state.LinkTimeout) {
			state.LinkTimeout--;
			PicoCStateLinkTimeoutSet(&state.LinkTimeout);
		}
		if (!state.LinkTimeout) {
			// telemetry link is lost. end here.
			break;
		}
		vTaskDelay(1000);
	}
}

/* ported printf function for stdOut */
void pprintf(const char *format, ...)
{
	uint8_t buffer[128];
	va_list args;
	uint16_t len;
	uint16_t i;

	va_start(args, format);
	vsprintf((char *)buffer, format, args);

	len = strlen((char *)buffer);
	if (len) {
		for (i=0; i<len; i++) {
			putch(buffer[i]);
		}
	}
}


/**
 * PicoC platform depending system functions
 * normaly stored in platform_xxx.c
 */

#ifdef NO_DEBUGGER
void DebugCleanup(Picoc *pc)
{
	// XXX - no debugger here
}
#endif

void PlatformInit(Picoc *pc)
{
}

void PlatformCleanup(Picoc *pc)
{
}

/* get a line of interactive input */
char *PlatformGetLine(char *line, int length, const char *prompt)
{
	int ix = 0;
	char *cp = line;
	char ch;

	pprintf("\n%s", prompt);
	length -= 2;

	while (1)
	{
		ch = getch();
		if (ch == 0x08)
		{	// Backspace pressed
			if (ix > 0)
			{
				putch(ch);
				putch(' ');
				putch(ch);
				--ix;
				--cp;
			}
			continue;
		}
		if (ch == 0x1B || ch == 0x03)
		{	// ESC character or Ctrl-C - exit
			pprintf("\nLeaving PicoC\n");
			break;
		}

		if (ix < length)
		{
			if (ch == '\r' || ch == '\n')
			{
				*cp++ = '\n';  // if newline, send newline character followed by null
				*cp = 0;
				putch('\n');
				return line;
			}
			*cp++ = ch;
			ix++;
			putch(ch);
		}
		else
		{
			pprintf("\n Line too long");
			pprintf("\n%s", prompt);
			ix = 0;
			cp = line;
		}
	}
	return NULL;
}

/* get a character of interactive input*/
int PlatformGetCharacter()
{
	return getch();
}

/* write a character to the console */
void PlatformPutc(unsigned char OutCh, union OutputStreamInfo *Stream)
{
	if (OutCh == '\n')	// send CRLF as CR
		putch('\r');
	putch(OutCh);
}

/* read and scan a file for definitions */
void PicocPlatformScanFile(Picoc *pc, const char *FileName)
{
	// XXX - unimplemented so far
}

/* exit the program */
void PlatformExit(Picoc *pc, int RetVal)
{
	pc->PicocExitValue = RetVal;
	longjmp(picocExitBuf, 1);
}


/**
 * PicoC platform depending library functions for taulabs
 * normaly stored in library_xxx.c
 */

/* testput(int): for simple program debug or store a value */
 void CtestPut (struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs) 
{
	int16_t val = Param[0]->Val->Integer;
	PicoCStateTestValueSet(&val);
}

/* testget(): for simple program debug or store a value */
 void CtestGet (struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs) 
{
	int16_t val; PicoCStateTestValueGet(&val);
	ReturnValue->Val->Integer = val;
}

/* delay(int): sleep for given ms-value */
 void Cdelay (struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs) 
{
	vTaskDelay(Param[0]->Val->Integer);
}

/* list of all library functions and their prototypes */
struct LibraryFunction PlatformLibrary[] =
{
	{ CtestPut, "void testPut(int);"},
	{ CtestGet, "int testGet();"},
	{ Cdelay,	"void delay(int);"},
	{ NULL, NULL }
};

void PlatformLibrarySetup(Picoc *pc)
{
}

void PlatformLibraryInit(Picoc *pc)
{
    IncludeRegister(pc, "taulabs.h", &PlatformLibrarySetup, &PlatformLibrary[0], NULL);
}

#endif /* PIOS_INCLUDE_PICOC */

/**
 * @}
 * @}
 */
