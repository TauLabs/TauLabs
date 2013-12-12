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
#ifdef __cplusplus
#define PI 3.14159265358979323846264338327950288 // [-]
#define DEG2RAD (PI / 180)
#define RAD2DEG (180 / PI)
#else
#define PI 3.14159265358979323846f // [-]
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)
#endif

// Temperature and pressure conversions
#define CELSIUS2KELVIN     273.15f
#define INCHES_MERCURY2KPA  3.386f
#define KPA2HECTAPASCAL     0.001f
#define HECTAPASCAL2KPA     100.0f

// Distance conversions
#define FEET2MILES                       0.3048f
#define KNOTS2M_PER_SECOND          0.514444444f
#define FEET_PER_SECOND2CM_PER_SECOND     30.48f
#define METERS_PER_SECOND2KM_PER_HOUR       3.6f
#define KM_PER_HOUR2METERS_PER_SECOND (1.0f/3.6f)
#define NM2DEG_LAT                         60.0f  // 60 nautical miles per degree latitude
#define DEG_LAT2NM                     (1.0/60.f) // 1 degree latitude per 60 nautical miles

// Standard atmospheric constants
#define UNIVERSAL_GAS_CONSTANT           8.31447f // [J/(mol·K)]
#define DRY_AIR_CONSTANT                 287.058f // [J/(kg*K)]
#define STANDARD_AIR_DENSITY               1.225f // [kg/m^3]
#define STANDARD_AIR_LAPSE_RATE           0.0065f // [deg/m]
#define STANDARD_AIR_MOLS2KG           0.0289644f // [kg/mol]
#define STANDARD_AIR_RELATIVE_HUMIDITY      20.0f // [%]
#define STANDARD_AIR_SEA_LEVEL_PRESSURE 101325.0f // [Pa]
#define STANDARD_AIR_TEMPERATURE (15.0f + CELSIUS2KELVIN) // Standard temperatue in [K]
#define STANDARD_AIR_MACH_SPEED           340.27f //speed of sound at standard sea level in [m/s]

// WGS-84 definitions (from http://home.online.no/~sigurdhu/WGS84_Eng.html)
#define WGS84_RADIUS_EARTH_KM          6371.008f  // Earth's radius in km
#define WGS84_A                        6378.137f  // semi-major axis of the ellipsoid in km
#define WGS84_B                    6356.7523142f  // semi-minor axis of the ellipsoid in km
#define WGS84_FLATTENING     3.35281066474748e-3f // flattening, i.e. (1 / 298.257223563)
#define WGS84_EPS             8.1819190842622e-2f // first eccentricity, i.e. sqrtf(1-WGS84_B^2/WGS84_A^2)
#define WGS84_EPS2                6.694379990e-3f // first eccentricity squared, i.e. WGS84_EPS^2

// Magnetic model parameters (from http://www.ngdc.noaa.gov/geomag/WMM/wmm_ddownload.shtml)
// Note: MUST be periodically updated. Please also update WorldMagneticModel.c and worldmagneticmodel.c at same time
// Last update is good until 2015 at the latest
#define MAGNETIC_MODEL_EDITION_DATE  5.7863328170559505e-307 
#define MAGNETIC_MODEL_EPOCH         2010.0f
#define MAGNETIC_MODEL_NAME          "WMM-2010"

#define COEFFS_FROM_NASA { {0, 0, 0, 0, 0, 0}, \
	{1, 0, -29496.6, 0.0, 11.6, 0.0}, \
	{1, 1, -1586.3, 4944.4, 16.5, -25.9}, \
	{2, 0, -2396.6, 0.0, -12.1, 0.0}, \
	{2, 1, 3026.1, -2707.7, -4.4, -22.5}, \
	{2, 2, 1668.6, -576.1, 1.9, -11.8}, \
	{3, 0, 1340.1, 0.0, 0.4, 0.0}, \
	{3, 1, -2326.2, -160.2, -4.1, 7.3}, \
	{3, 2, 1231.9, 251.9, -2.9, -3.9}, \
	{3, 3, 634.0, -536.6, -7.7, -2.6}, \
	{4, 0, 912.6, 0.0, -1.8, 0.0}, \
	{4, 1, 808.9, 286.4, 2.3, 1.1}, \
	{4, 2, 166.7, -211.2, -8.7, 2.7}, \
	{4, 3, -357.1, 164.3, 4.6, 3.9}, \
	{4, 4, 89.4, -309.1, -2.1, -0.8}, \
	{5, 0, -230.9, 0.0, -1.0, 0.0}, \
	{5, 1, 357.2, 44.6, 0.6, 0.4}, \
	{5, 2, 200.3, 188.9, -1.8, 1.8}, \
	{5, 3, -141.1, -118.2, -1.0, 1.2}, \
	{5, 4, -163.0, 0.0, 0.9, 4.0}, \
	{5, 5, -7.8, 100.9, 1.0, -0.6}, \
	{6, 0, 72.8, 0.0, -0.2, 0.0}, \
	{6, 1, 68.6, -20.8, -0.2, -0.2}, \
	{6, 2, 76.0, 44.1, -0.1, -2.1}, \
	{6, 3, -141.4, 61.5, 2.0, -0.4}, \
	{6, 4, -22.8, -66.3, -1.7, -0.6}, \
	{6, 5, 13.2, 3.1, -0.3, 0.5}, \
	{6, 6, -77.9, 55.0, 1.7, 0.9}, \
	{7, 0, 80.5, 0.0, 0.1, 0.0}, \
	{7, 1, -75.1, -57.9, -0.1, 0.7}, \
	{7, 2, -4.7, -21.1, -0.6, 0.3}, \
	{7, 3, 45.3, 6.5, 1.3, -0.1}, \
	{7, 4, 13.9, 24.9, 0.4, -0.1}, \
	{7, 5, 10.4, 7.0, 0.3, -0.8}, \
	{7, 6, 1.7, -27.7, -0.7, -0.3}, \
	{7, 7, 4.9, -3.3, 0.6, 0.3}, \
	{8, 0, 24.4, 0.0, -0.1, 0.0}, \
	{8, 1, 8.1, 11.0, 0.1, -0.1}, \
	{8, 2, -14.5, -20.0, -0.6, 0.2}, \
	{8, 3, -5.6, 11.9, 0.2, 0.4}, \
	{8, 4, -19.3, -17.4, -0.2, 0.4}, \
	{8, 5, 11.5, 16.7, 0.3, 0.1}, \
	{8, 6, 10.9, 7.0, 0.3, -0.1}, \
	{8, 7, -14.1, -10.8, -0.6, 0.4}, \
	{8, 8, -3.7, 1.7, 0.2, 0.3}, \
	{9, 0, 5.4, 0.0, 0.0, 0.0}, \
	{9, 1, 9.4, -20.5, -0.1, 0.0}, \
	{9, 2, 3.4, 11.5, 0.0, -0.2}, \
	{9, 3, -5.2, 12.8, 0.3, 0.0}, \
	{9, 4, 3.1, -7.2, -0.4, -0.1}, \
	{9, 5, -12.4, -7.4, -0.3, 0.1}, \
	{9, 6, -0.7, 8.0, 0.1, 0.0}, \
	{9, 7, 8.4, 2.1, -0.1, -0.2}, \
	{9, 8, -8.5, -6.1, -0.4, 0.3}, \
	{9, 9, -10.1, 7.0, -0.2, 0.2}, \
	{10, 0, -2.0, 0.0, 0.0, 0.0}, \
	{10, 1, -6.3, 2.8, 0.0, 0.1}, \
	{10, 2, 0.9, -0.1, -0.1, -0.1}, \
	{10, 3, -1.1, 4.7, 0.2, 0.0}, \
	{10, 4, -0.2, 4.4, 0.0, -0.1}, \
	{10, 5, 2.5, -7.2, -0.1, -0.1}, \
	{10, 6, -0.3, -1.0, -0.2, 0.0}, \
	{10, 7, 2.2, -3.9, 0.0, -0.1}, \
	{10, 8, 3.1, -2.0, -0.1, -0.2}, \
	{10, 9, -1.0, -2.0, -0.2, 0.0}, \
	{10, 10, -2.8, -8.3, -0.2, -0.1}, \
	{11, 0, 3.0, 0.0, 0.0, 0.0}, \
	{11, 1, -1.5, 0.2, 0.0, 0.0}, \
	{11, 2, -2.1, 1.7, 0.0, 0.1}, \
	{11, 3, 1.7, -0.6, 0.1, 0.0}, \
	{11, 4, -0.5, -1.8, 0.0, 0.1}, \
	{11, 5, 0.5, 0.9, 0.0, 0.0}, \
	{11, 6, -0.8, -0.4, 0.0, 0.1}, \
	{11, 7, 0.4, -2.5, 0.0, 0.0}, \
	{11, 8, 1.8, -1.3, 0.0, -0.1}, \
	{11, 9, 0.1, -2.1, 0.0, -0.1}, \
	{11, 10, 0.7, -1.9, -0.1, 0.0}, \
	{11, 11, 3.8, -1.8, 0.0, -0.1}, \
	{12, 0, -2.2, 0.0, 0.0, 0.0}, \
	{12, 1, -0.2, -0.9, 0.0, 0.0}, \
	{12, 2, 0.3, 0.3, 0.1, 0.0}, \
	{12, 3, 1.0, 2.1, 0.1, 0.0}, \
	{12, 4, -0.6, -2.5, -0.1, 0.0}, \
	{12, 5, 0.9, 0.5, 0.0, 0.0}, \
	{12, 6, -0.1, 0.6, 0.0, 0.1}, \
	{12, 7, 0.5, 0.0, 0.0, 0.0}, \
	{12, 8, -0.4, 0.1, 0.0, 0.0}, \
	{12, 9, -0.4, 0.3, 0.0, 0.0}, \
	{12, 10, 0.2, -0.9, 0.0, 0.0}, \
	{12, 11, -0.8, -0.2, -0.1, 0.0}, \
	{12, 12, 0.0, 0.9, 0.1, 0.0}}

#endif /* PHYSICAL_CONSTANTS_H_ */

/**
 * @}
 * @}
 */
