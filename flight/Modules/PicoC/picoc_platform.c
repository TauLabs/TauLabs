/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup PicoC Interpreter Module
 * @{ 
 *
 * @file       picoc_platform.c
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
#include "picocstatus.h"
#include "picoc_port.h"
#include <setjmp.h>

// Private variables
jmp_buf picocExitBuf;

/**
 * picoc implemetation
 * This is needed to compile picoc without a makefile to prevent object name conflicts.
 * All original picoc source files are in the include folder.
 */
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

// Private functions
uint8_t getch();
void putch(uint8_t ch);
void pprintf(const char *format, ...);


/**
 * picoc main program
 * parses source or switches to interactive mode
 * returns the exit() value
 */
int picoc(const char * source, int stack_size)
{
	Picoc pc;
	PicocInitialise(&pc, stack_size);

	if (PicocPlatformSetExitPoint(&pc)) {
		// we get here, if an error occures or 'exit();' was called.
		PicocCleanup(&pc);
		return pc.PicocExitValue;
	}

	if (source) {
		PicocParse(&pc, "nofile", source, strlen(source), TRUE, TRUE, FALSE, FALSE);
	} else {
		PicocParseInteractive(&pc);
	}

	PicocCleanup(&pc);
	return pc.PicocExitValue;
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
 * basic functions for virtual stdIO
 * uses a optional USART or telemetry tunnel for communication
 */

/* get a character from stdIn or telemetry link */
uint8_t getch()
{
	uint8_t ch = 0;
	uintptr_t stdIO = 0;
	uint8_t stdIn;
	uint16_t timeout;

#ifdef PIOS_COM_PICOC
	stdIO = PIOS_COM_PICOC;
#endif

	while (1) {
		// try to get a char from stdIO
		if (stdIO) {
			if (PIOS_COM_ReceiveBuffer(stdIO, &ch, 1, 0) == 1) {
				break;
			}
		}
		// if there is a telemetry link, try it
		PicoCStatusLinkTimeoutGet(&timeout);
		PicoCStatusStdInGet(&stdIn);
		if ((timeout) && (stdIn)) {
			ch = stdIn;
			stdIn = 0;
			PicoCStatusStdInSet(&stdIn);
			break;
		}
		if (timeout) {
			timeout--;
			PicoCStatusLinkTimeoutSet(&timeout);
		}
		if ((!stdIO) && (!timeout)) {
			// there is no stdIO and telemetry link is lost. end here.
			ch = 0;
			break;
		}
		vTaskDelay(1);
	}
	return ch;
}

/* put a character to stdOut */
void putch(uint8_t ch)
{
	uintptr_t stdIO = 0;
	uint8_t stdOut;
	uint16_t timeout;

#ifdef PIOS_COM_PICOC
	stdIO = PIOS_COM_PICOC;
#endif
	if (stdIO) {
		PIOS_COM_SendChar(stdIO, ch);
	}

	// if there is a telemetrylink, send it to gcs too.
	while(1) {
		PicoCStatusLinkTimeoutGet(&timeout);
		PicoCStatusStdOutGet(&stdOut);
		if ((timeout) && (!stdOut)) {
			PicoCStatusStdOutSet(&ch);
			break;
		}
		if (timeout) {
			timeout--;
			PicoCStatusLinkTimeoutSet(&timeout);
		}
		if (!timeout) {
			// telemetry link is lost. end here.
			break;
		}
		vTaskDelay(1);
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


#endif /* PIOS_INCLUDE_PICOC */


/**
 * @}
 * @}
 */
