/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_OVERO Overo Functions
 * @{
 *
 * @file       pios_overo.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @brief      Overo functions header.
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

#ifndef PIOS_OVERO_H
#define PIOS_OVERO_H

extern void PIOS_OVERO_DMA_irq_handler(uint32_t overo_id);
extern int32_t PIOS_OVERO_GetPacketCount(uint32_t overo_id);
extern int32_t PIOS_OVERO_GetWrittenBytes(uint32_t overo_id);
extern int32_t PIOS_OVERO_Enable(uint32_t overo_id);
extern int32_t PIOS_OVERO_Disable(uint32_t overo_id);

#endif /* PIOS_OVERO_H */

/**
 * @}
 * @}
 */
