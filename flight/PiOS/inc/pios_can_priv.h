/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_CAN PiOS CAN interface layer
 * @brief CAN interface for PiOS
 * @{
 *
 * @file       pios_can_priv.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      PiOS CAN interface header
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

#if !defined(PIOS_CAN_PRIV_H)
#define PIOS_CAN_PRIV_H

extern const struct pios_com_driver pios_can_com_driver;

struct pios_can_cfg {
	CAN_TypeDef *regs;          //! CAN device to configure
	CAN_InitTypeDef init;       //! Init config for CAN device
	uint32_t remap;             //! GPIO remapping to alternative function
	struct stm32_gpio rx;       //! Configuration for RX pin
	struct stm32_gpio tx;       //! Configuration for TX pin
	struct stm32_irq rx_irq;    //! Configuration for IRQ
	struct stm32_irq tx_irq;    //! Configuration for IRQ
};

/**
 * Initialize the CAN driver and return an opaque id
 * @param[out]   id the CAN interface handle
 * @param[in]    cfg the configuration structure
 * @return 0 if successful, negative otherwise
 */
int32_t PIOS_CAN_Init(uintptr_t *id, const struct pios_can_cfg *cfg);

#endif /* PIOS_CAN_PRIV_H */

/**
 * @}
 * @}
 */
 
