/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup AirspeedModule Airspeed Module
 * @{ 
 *
 * @file       baro_airspeed_analog.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      Read airspeed from analog sensor
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
#ifndef ANALOG_AIRSPEED_H
#define ANALOG_AIRSPEED_H

void baro_airspeedGetAnalog(BaroAirspeedData *baroAirspeedData, uint32_t *lastSysTime, uint8_t airspeedSensorType, int8_t airspeedADCPin);

#endif // ANALOG_AIRSPEED_H

/**
 * @}
 * @}
 */
