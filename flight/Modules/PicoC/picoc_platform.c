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
 *             platform function library for picoc
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
#include "picocstatus.h"
#include <setjmp.h>

// Private variables
static char *heap_memory;
static size_t heap_size;
static bool heap_used;
jmp_buf PicocExitBuf;

/**
 * picoc implemetation
 * This is needed to compile picoc without a makefile to prevent object name conflicts.
 * All original picoc source files are in the include folder.
 */
#include "inc/table.c"
#include "inc/lex.c"
#include "inc/parse.c"
#include "inc/expression.c"
#include "inc/heap.c"
#include "inc/type.c"
#include "inc/variable.c"
//#include "inc/clibrary.c"	/* we have our own optimized version. */
#include "inc/platform.c"
#include "inc/include.c"
#include "inc/debug.c"

/**
 * picoc main program
 * parses source or switches to interactive mode
 * returns the exit() value
 */
int picoc(const char * source, size_t stack_size)
{
	Picoc pc;
	PicocInitialise(&pc, stack_size);

	if (PicocPlatformSetExitPoint(&pc))
	{	/* we get here, if an error occures or 'exit();' was called. */
		PicocCleanup(&pc);
		return pc.PicocExitValue;
	}

	if (source)
	{	/* start with complete source file */
		PicocParse(&pc, "nofile", source, strlen(source), true, true, false, false);
	}
	else
	{	/* start interactive */
		PicocParseInteractive(&pc);
	}

	PicocCleanup(&pc);
	return pc.PicocExitValue;
}

/**
 * PicoC platform depending system functions
 * normaly stored in platform_xxx.c
 */
void PlatformInit(Picoc *pc)
{
}

void PlatformCleanup(Picoc *pc)
{
}

/* get a line of interactive input */
char *PlatformGetLine(char *line, int length, const char *prompt)
{
#ifdef PIOS_COM_PICOC
	if (PIOS_COM_PICOC == 0)
		return NULL;

	int ix = 0;
	char *cp = line;
	uint8_t ch;

	PIOS_COM_SendFormattedString(PIOS_COM_PICOC, "\n%s", prompt);
	length -= 2;

	while (1)
	{
		ch = 0;
		while (PIOS_COM_ReceiveBuffer(PIOS_COM_PICOC, &ch, 1, 10) == 0);

		if (ch == '\b')
		{	// Backspace pressed
			if (ix > 0)
			{
				PIOS_COM_SendString(PIOS_COM_PICOC, "\b \b");
				--ix;
				--cp;
			}
			continue;
		}

		if (ch == 0x1b || ch == 0x03)
		{	// ESC character or Ctrl-C - exit
			PIOS_COM_SendString(PIOS_COM_PICOC, "\nLeaving PicoC\n");
			break;
		}

		if (ix < length)
		{
			if (ch == '\r' || ch == '\n')
			{
				*cp++ = '\n'; // if newline, send newline character followed by null
				*cp = 0;
				PIOS_COM_SendChar(PIOS_COM_PICOC, '\n');
				return line;
			}
			*cp++ = ch;
			ix++;
			PIOS_COM_SendChar(PIOS_COM_PICOC, ch);
		}
		else
		{
			PIOS_COM_SendFormattedString(PIOS_COM_PICOC, "\nLine too long\n%s", prompt);
			ix = 0;
			cp = line;
		}
	}
#endif
	return NULL;
}

/* get a character of interactive input */
int PlatformGetCharacter()
{
#ifdef PIOS_COM_PICOC
	uint8_t ch = 0;
	if ((PIOS_COM_PICOC) && (PIOS_COM_ReceiveBuffer(PIOS_COM_PICOC, &ch, 1, 0) == 1))
		return ch;
#endif
	return -1;
}

/* write a character to the console */
void PlatformPutc(unsigned char OutCh, union OutputStreamInfo *Stream)
{
#ifdef PIOS_COM_PICOC
	PIOS_COM_SendChar(PIOS_COM_PICOC, OutCh);
#endif
}

/* read and scan a file for definitions */
void PicocPlatformScanFile(Picoc *pc, const char *FileName)
{
	PlatformDebug("no filesystem. %s failed", FileName);
}

/* exit the program */
void PlatformExit(Picoc *pc, int RetVal)
{
	pc->PicocExitValue = RetVal;
	longjmp(PicocExitBuf, 1);
}

/**
 * simple memory functions
 * On our platform free() is not functional. This is a workaround for this.
 */
void *PlatformMalloc(size_t size)
{
	if (heap_memory == NULL)
	{	/* no heap memory used yet. try to get some */
		heap_size = size;
		heap_memory = (char *)pvPortMalloc(heap_size);
	}
	if ((heap_size >= size) && (heap_memory != NULL) && (!heap_used))
	{	/* memory is free and size fits */
		heap_used = true;
		return heap_memory;
	}
	return NULL;
}

void PlatformFree(void *ptr)
{
	if (ptr == heap_memory)
	{	/* fake free() */
		heap_used = false;
	}
}

size_t PlatformHeapSize()
{	/* a replacement for #define HEAP_SIZE. So it is variable */
	return heap_size;
}

/* a little debug message function */
void PlatformDebug(const char *format, ...)
{
#ifdef PIOS_COM_PICOC
	uint8_t buffer[128];
	va_list args;

	va_start(args, format);
	vsprintf((char *)buffer, format, args);

	PIOS_COM_SendFormattedString(PIOS_COM_PICOC, "[debug:%s]\n", buffer);
	vTaskDelay(200); // delay to make sure, this is sent out.
#endif
}

#endif /* PIOS_INCLUDE_PICOC */

/**
 * @}
 * @}
 */
