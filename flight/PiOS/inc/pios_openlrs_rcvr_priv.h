/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_RFM22b RFM22b receiver functions
 * @brief Deals with the RFM22b module
 * @{
 *
 * @file       pios_rfm22b_rcvr_priv.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2013.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      TauLabs Link receiver private functions
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

#ifndef PIOS_OPENLRS_RCVR_PRIV_H
#define PIOS_OPENLRS_RCVR_PRIV_H

#include <pios.h>

extern const struct pios_rcvr_driver pios_openlrs_rcvr_driver;

extern int32_t PIOS_OpenLRS_Rcvr_Init(uintptr_t * openlrs_rcvr_id, uintptr_t openlrs_id);
extern int32_t PIOS_OpenLRS_Rcvr_UpdateChannels(uintptr_t openlrs_rcvr_id, int16_t * channels);

#endif /* PIOS_OPENLRS_RCVR_PRIV_H */

/**
 * @}
 * @}
 */
