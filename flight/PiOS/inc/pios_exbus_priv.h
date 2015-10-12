/**
 ******************************************************************************
 * @file       pios_exbus_priv.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_EXBUS Jeti EX Bus receiver functions
 * @{
 * @brief Jeti EX Bus receiver functions
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

#ifndef PIOS_EXBUS_PRIV_H
#define PIOS_EXBUS_PRIV_H

#include <pios.h>
#include <pios_usart_priv.h>


/* EXBUS receiver instance configuration */
extern const struct pios_rcvr_driver pios_exbus_rcvr_driver;

extern int32_t PIOS_EXBUS_Init(uintptr_t *exbus_id,
			     const struct pios_com_driver *driver,
			     uintptr_t lower_id);

#endif /* PIOS_EXBUS_PRIV_H */

/**
 * @}
 * @}
 */
