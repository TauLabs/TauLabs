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

// defines for picoc
#define HEAP_SIZE (16*1024)
//#define USE_MALLOC_STACK                   /* stack is allocated using malloc() */
//#define USE_MALLOC_HEAP                    /* heap is allocated using malloc() */
#define BUILTIN_MINI_STDLIB
#define PICOC_LIBRARY
#define NO_CTYPE
#define NO_DEBUGGER
#define NO_CALLOC
#define NO_REALLOC
#define NO_STRING_FUNCTIONS
#define malloc pvPortMalloc
#define calloc(a,b) pvPortMalloc(a*b)
#define free vPortFree
#define assert(x)
#define PicocPlatformSetExitPoint(pc) setjmp(picocExitBuf)

#include "picoc.h"

int picoc(const char * source, int stack_size);
