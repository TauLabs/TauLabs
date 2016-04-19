/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Rate Torque Linear Quadratic Gaussian controller
 * @{
 *
 * @file       rate_torque_si.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      System identication of parameters of this model
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************/
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

#ifndef RATE_TORQUE_SI_H
#define RATE_TORQUE_SI_H

bool rtsi_alloc(uintptr_t *rtsi_state);
void rtsi_predict(uintptr_t rtsi_handle, const float u_in[3], const float gyro[3], const float dT_s);
void rtsi_init(uintptr_t rtsi_handle);

void rtsi_get_rates(uintptr_t rtsi_handle, float *rates);
void rtsi_get_torque(uintptr_t rtsi_handle, float *torque);
void rtsi_get_gains(uintptr_t rtsi_handle, float *gains);
void rtsi_get_bias(uintptr_t rtsi_handle, float *bias);
void rtsi_get_tau(uintptr_t rtsi_handle, float *tau);

#endif /* RATE_TORQUE_SI_H */
 
 /**
 * @}
 * @}
 */
