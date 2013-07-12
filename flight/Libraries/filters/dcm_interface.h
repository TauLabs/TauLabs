/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @file       dcm_interface.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Interface from the SE(3)+ infrastructure to DCM
 *
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

#if !defined(DCM_INTERFACE_H)
#define DCM_INTERFACE_H

#include "filter_interface.h"
#include "filter_infrastructure_se3.h"

extern struct filter_driver dcm_filter_driver;

#endif /* DCM_INTERFACE_H */