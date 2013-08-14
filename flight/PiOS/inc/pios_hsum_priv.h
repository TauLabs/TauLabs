/**
 ******************************************************************************
 * @file       pios_hsum.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_HSUM Graupner HoTT receiver functions
 * @{
 * @brief Graupner HoTT receiver functions for SUMD/H
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

#ifndef PIOS_HSUM_PRIV_H
#define PIOS_HSUM_PRIV_H

#include <pios.h>
#include <pios_usart_priv.h>

/* HSUM protocol variations */
enum pios_hsum_proto {
	PIOS_HSUM_PROTO_SUMD,
	PIOS_HSUM_PROTO_SUMH,
};

/* HSUM receiver instance configuration */
extern const struct pios_rcvr_driver pios_hsum_rcvr_driver;

extern int32_t PIOS_HSUM_Init(uintptr_t *hsum_id,
			     const struct pios_com_driver *driver,
			     uintptr_t lower_id,
			     enum pios_hsum_proto proto);

#endif /* PIOS_HSUM_PRIV_H */

/**
 * @}
 * @}
 */
