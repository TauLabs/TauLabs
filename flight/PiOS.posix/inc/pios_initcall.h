/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Initcall infrastructure
 * @{
 * @addtogroup   PIOS_INITCALL Generic Initcall Macros
 * @brief Initcall Macros
 * @{
 *
 * @file       pios_initcall.h  
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Initcall header
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

#ifndef PIOS_INITCALL_H
#define PIOS_INITCALL_H

typedef int32_t (*initcall_t)(void);
typedef struct {
	initcall_t fn_minit;
	initcall_t fn_tinit;
} initmodule_t;

/* Init module section */
extern initmodule_t *__module_initcall_start, *__module_initcall_end;

#define MODULE_INITCALL(ifn, sfn) \
static void _add_init_fn(void) __attribute__((constructor)); \
static void _add_init_fn(void) { \
	__module_initcall_end->fn_minit = (ifn); \
	__module_initcall_end->fn_tinit = (sfn); \
	__module_initcall_end++; \
}

#define MODULE_INITSYSTEM_DECLS \
static initmodule_t __module_initcalls[256]; \
initmodule_t *__module_initcall_start = __module_initcalls; \
initmodule_t *__module_initcall_end = __module_initcalls;

#define MODULE_INITIALISE_ALL(wdgfn)  { \
	for (initmodule_t *fn = __module_initcall_start; fn < __module_initcall_end; fn++) { \
		if (fn->fn_minit)                               \
		(fn->fn_minit)();                       \
		(wdgfn)();                                      \
	}                                                       \
}

#define MODULE_TASKCREATE_ALL  { for (initmodule_t *fn = __module_initcall_start; fn < __module_initcall_end; fn++) \
	if (fn->fn_tinit) \
	(fn->fn_tinit)(); }


#endif	/* PIOS_INITCALL_H */

/**
 * @}
 * @}
 */
