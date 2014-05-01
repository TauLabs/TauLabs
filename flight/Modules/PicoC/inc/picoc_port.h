/**
 ******************************************************************************
  * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup PicoC Interpreter Module
 * @{ 
 *
 * @file       picocmodule.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      c-interpreter module for autonomous user programmed tasks
 *             header file to 
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

/* defines for picoc */
#ifndef PICOC_PORT_H
#define PICOC_PORT_H

/* used picoc subversion number */
#define VER "603"

/* host specific platform defines */
#define HEAP_SIZE PlatformHeapSize()
#define USE_MALLOC_STACK
#define NO_CTYPE
#define NO_DEBUGGER
#define BUILTIN_MINI_STDLIB
//#define NO_FP
//#define NO_STRING_FUNCTIONS
#define assert PIOS_Assert
#define malloc PlatformMalloc
#define free PlatformFree
#define PicocPlatformSetExitPoint(pc) setjmp(PicocExitBuf)

/* function prototypes */
void *PlatformMalloc(size_t size);
void PlatformFree(void *ptr);
size_t PlatformHeapSize();
void PlatformDebug(const char *format, ...);
int picoc(const char *source, size_t stack_size);

/* get all picoc definitions */
#include "picoc.h"

/* add missing things */
extern struct LibraryFunction CLibrary[];

#ifdef NO_CTYPE
#define isdigit(c) ((c) >= '0' && (c) <= '9')
#endif

#ifdef NO_DEBUGGER
#define DebugInit(pc)
#define DebugCleanup(pc)
#define DebugCheckStatement(parser)
#endif

#endif /* PICOC_PORT_H */

/**
 * @}
 * @}
 */
