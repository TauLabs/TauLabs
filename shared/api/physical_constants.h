/**
 ******************************************************************************
 * @file       physical_constants.h
 * @author     Tau Labs, http://ww.taulabs.org, Copyright (C) 2013
 * @addtogroup Physical constants
 * @{
 * @addtogroup 
 * @{
 * @brief A file where physical constants can be placed.
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

#ifndef PHYSICAL_CONSTANTS_H_
#define PHYSICAL_CONSTANTS_H_

// Physical constants
#define GRAVITY 9.805f // [m/s^2]

// Trigonometry
#define PI 3.14159265358979323846f // [-]
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

// Unit conversions
#define CELSIUS2KELVIN               273.15f

// Standard atmospheric constants
#define UNIVERSAL_GAS_CONSTANT          8.31447f //[J/(mol·K)]
#define DRY_AIR_CONSTANT                287.058f //[J/(kg*K)]
#define STANDARD_AIR_DENSITY              1.225f //[kg/m^3]
#define STANDARD_AIR_LAPSE_RATE          0.0065f //[deg/m]
#define STANDARD_AIR_MOLS2KG          0.0289644f //[kg/mol]
#define STANDARD_AIR_RELATIVE_HUMIDITY       20f //[%]
#define STANDARD_AIR_SEA_LEVEL_PRESSURE 101.325f //[kPa]
#define STANDARD_AIR_TEMPERATURE (15.0f + CELSIUS2KELVIN) // Standard temperatue in [K]

#endif /* PHYSICAL_CONSTANTS_H_ */

/**
 * @}
 * @}
 */
