/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_SYS System Functions
 * @brief PIOS System Initialization code
 * @{
 *
 * @file       pios_sys.c  
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * 	        Parts by Thorsten Klose (tk@midibox.org) (tk@midibox.org)
 * @brief      Sets up basic STM32 system hardware, functions are called from Main.
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

/* Project Includes */
#define _GNU_SOURCE
#include "pios.h"

#if defined(PIOS_INCLUDE_SYS)


/**
* Initialises all system peripherals
*/
#include <assert.h>		/* assert */
#include <stdlib.h>		/* printf */
#include <signal.h>		/* sigaction */
#include <fenv.h>		/* PE_* */
static void sigint_handler(int signum, siginfo_t *siginfo, void *ucontext)
{
	printf("\nSIGINT received.  Shutting down\n");
	exit(0);
}

static void sigfpe_handler(int signum, siginfo_t *siginfo, void *ucontext)
{
	printf("\nSIGFPE received.  OMG!  Math Bug!  Run again with gdb to find your mistake.\n");
	exit(0);
}

void PIOS_SYS_Init(void)
{
	struct sigaction sa_int = {
		.sa_sigaction = sigint_handler,
		.sa_flags = SA_SIGINFO,
	};

	int rc = sigaction(SIGINT, &sa_int, NULL);
	assert(rc == 0);

	struct sigaction sa_fpe = {
		.sa_sigaction = sigfpe_handler,
		.sa_flags = SA_SIGINFO,
	};

	rc = sigaction(SIGFPE, &sa_fpe, NULL);
	assert(rc == 0);

	feenableexcept(FE_DIVBYZERO | FE_UNDERFLOW | FE_OVERFLOW | FE_INVALID);
}

/**
* Shutdown PIOS and reset the microcontroller:<BR>
* <UL>
*   <LI>Disable all RTOS tasks
*   <LI>Disable all interrupts
*   <LI>Turn off all board LEDs
*   <LI>Reset STM32
* </UL>
* \return < 0 if reset failed
*/
int32_t PIOS_SYS_Reset(void)
{
	/* We will never reach this point */
	return -1;
}

/**
* Returns the CPU's flash size (in bytes)
*/
uint32_t PIOS_SYS_getCPUFlashSize(void)
{
	return 1024000;
}

/**
* Returns the serial number as a string
* param[out] uint8_t pointer to a string which can store at least 12 bytes
* (12 bytes returned for STM32)
* return < 0 if feature not supported
*/
int32_t PIOS_SYS_SerialNumberGetBinary(uint8_t *array)
{
	/* Stored in the so called "electronic signature" */
	for (int i = 0; i < PIOS_SYS_SERIAL_NUM_BINARY_LEN; ++i) {
		array[i] = 0xff;
	}

	/* No error */
	return 0;
}

/**
* Returns the serial number as a string
* param[out] str pointer to a string which can store at least 32 digits + zero terminator!
* (24 digits returned for STM32)
* return < 0 if feature not supported
*/
int32_t PIOS_SYS_SerialNumberGet(char *str)
{
	/* Stored in the so called "electronic signature" */
	int i;
	for (i = 0; i < PIOS_SYS_SERIAL_NUM_ASCII_LEN; ++i) {
		str[i] = 'F';
	}
	str[i] = '\0';

	/* No error */
	return 0;
}

#endif

/**
  * @}
  * @}
  */
