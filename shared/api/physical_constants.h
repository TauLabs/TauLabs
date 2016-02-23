/**
 ******************************************************************************
 * @file       physical_constants.h
 * @author     Tau Labs, http://ww.taulabs.org, Copyright (C) 2013-2016
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

#ifndef M_PI
#define M_PI PI
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
#define FEET_PER_SECOND2M_PER_SECOND     0.3048f
#define METERS_PER_SECOND2KM_PER_HOUR       3.6f
#define KM_PER_HOUR2METERS_PER_SECOND (1.0f/3.6f)
#define NM2DEG_LAT                         60.0f  // 60 nautical miles per degree latitude
#define DEG_LAT2NM                     (1.0/60.f) // 1 degree latitude per 60 nautical miles
#define MS_TO_KMH                      METERS_PER_SECOND2KM_PER_HOUR
#define MS_TO_MPH                      2.23694f
#define M_TO_FEET                      3.28084f
#define FEET_PER_MILE                  5280.f

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
// Last update is good until 2020 at the latest
#define MAGNETIC_MODEL_EDITION_DATE  5.7863328170559505e-307 
#define MAGNETIC_MODEL_EPOCH         2015.0f
#define MAGNETIC_MODEL_NAME          "WMM-2015"

#define COEFFS_FROM_NASA { {0, 0, 0, 0, 0, 0}, \
	{1, 0, -29438.5, 0.0, 10.7, 0.0}, \
	{1, 1, -1501.1, 4796.2, 17.9, -26.8}, \
	{2, 0, -2445.3, 0.0, -8.6, 0.0}, \
	{2, 1, 3012.5, -2845.6, -3.3, -27.1}, \
	{2, 2, 1676.6, -642.0, 2.4, -13.3}, \
	{3, 0, 1351.1, 0.0, 3.1, 0.0}, \
	{3, 1, -2352.3, -115.3, -6.2, 8.4}, \
	{3, 2, 1225.6, 245.0, -0.4, -0.4}, \
	{3, 3, 581.9, -538.3, -10.4, 2.3}, \
	{4, 0, 907.2, 0.0, -0.4, 0.0}, \
	{4, 1, 813.7, 283.4, 0.8, -0.6}, \
	{4, 2, 120.3, -188.6, -9.2, 5.3}, \
	{4, 3, -335.0, 180.9, 4.0, 3.0}, \
	{4, 4, 70.3, -329.5, -4.2, -5.3}, \
	{5, 0, -232.6, 0.0, -0.2, 0.0}, \
	{5, 1, 360.1, 47.4, 0.1, 0.4}, \
	{5, 2, 192.4, 196.9, -1.4, 1.6}, \
	{5, 3, -141.0, -119.4, 0.0, -1.1}, \
	{5, 4, -157.4, 16.1, 1.3, 3.3}, \
	{5, 5, 4.3, 100.1, 3.8, 0.1}, \
	{6, 0, 69.5, 0.0, -0.5, 0.0}, \
	{6, 1, 67.4, -20.7, -0.2, 0.0}, \
	{6, 2, 72.8, 33.2, -0.6, -2.2}, \
	{6, 3, -129.8, 58.8, 2.4, -0.7}, \
	{6, 4, -29.0, -66.5, -1.1, 0.1}, \
	{6, 5, 13.2, 7.3, 0.3, 1.0}, \
	{6, 6, -70.9, 62.5, 1.5, 1.3}, \
	{7, 0, 81.6, 0.0, 0.2, 0.0}, \
	{7, 1, -76.1, -54.1, -0.2, 0.7}, \
	{7, 2, -6.8, -19.4, -0.4, 0.5}, \
	{7, 3, 51.9, 5.6, 1.3, -0.2}, \
	{7, 4, 15.0, 24.4, 0.2, -0.1}, \
	{7, 5, 9.3, 3.3, -0.4, -0.7}, \
	{7, 6, -2.8, -27.5, -0.9, 0.1}, \
	{7, 7, 6.7, -2.3, 0.3, 0.1}, \
	{8, 0, 24.0, 0.0, 0.0, 0.0}, \
	{8, 1, 8.6, 10.2, 0.1, -0.3}, \
	{8, 2, -16.9, -18.1, -0.5, 0.3}, \
	{8, 3, -3.2, 13.2, 0.5, 0.3}, \
	{8, 4, -20.6, -14.6, -0.2, 0.6}, \
	{8, 5, 13.3, 16.2, 0.4, -0.1}, \
	{8, 6, 11.7, 5.7, 0.2, -0.2}, \
	{8, 7, -16.0, -9.1, -0.4, 0.3}, \
	{8, 8, -2.0, 2.2, 0.3, 0.0}, \
	{9, 0, 5.4, 0.0, 0.0, 0.0}, \
	{9, 1, 8.8, -21.6, -0.1, -0.2}, \
	{9, 2, 3.1, 10.8, -0.1, -0.1}, \
	{9, 3, -3.1, 11.7, 0.4, -0.2}, \
	{9, 4, 0.6, -6.8, -0.5, 0.1}, \
	{9, 5, -13.3, -6.9, -0.2, 0.1}, \
	{9, 6, -0.1, 7.8, 0.1, 0.0}, \
	{9, 7, 8.7, 1.0, 0.0, -0.2}, \
	{9, 8, -9.1, -3.9, -0.2, 0.4}, \
	{9, 9, -10.5, 8.5, -0.1, 0.3}, \
	{10, 0, -1.9, 0.0, 0.0, 0.0}, \
	{10, 1, -6.5, 3.3, 0.0, 0.1}, \
	{10, 2, 0.2, -0.3, -0.1, -0.1}, \
	{10, 3, 0.6, 4.6, 0.3, 0.0}, \
	{10, 4, -0.6, 4.4, -0.1, 0.0}, \
	{10, 5, 1.7, -7.9, -0.1, -0.2}, \
	{10, 6, -0.7, -0.6, -0.1, 0.1}, \
	{10, 7, 2.1, -4.1, 0.0, -0.1}, \
	{10, 8, 2.3, -2.8, -0.2, -0.2}, \
	{10, 9, -1.8, -1.1, -0.1, 0.1}, \
	{10, 10, -3.6, -8.7, -0.2, -0.1}, \
	{11, 0, 3.1, 0.0, 0.0, 0.0}, \
	{11, 1, -1.5, -0.1, 0.0, 0.0}, \
	{11, 2, -2.3, 2.1, -0.1, 0.1}, \
	{11, 3, 2.1, -0.7, 0.1, 0.0}, \
	{11, 4, -0.9, -1.1, 0.0, 0.1}, \
	{11, 5, 0.6, 0.7, 0.0, 0.0}, \
	{11, 6, -0.7, -0.2, 0.0, 0.0}, \
	{11, 7, 0.2, -2.1, 0.0, 0.1}, \
	{11, 8, 1.7, -1.5, 0.0, 0.0}, \
	{11, 9, -0.2, -2.5, 0.0, -0.1}, \
	{11, 10, 0.4, -2.0, -0.1, 0.0}, \
	{11, 11, 3.5, -2.3, -0.1, -0.1}, \
	{12, 0, -2.0, 0.0, 0.1, 0.0}, \
	{12, 1, -0.3, -1.0, 0.0, 0.0}, \
	{12, 2, 0.4, 0.5, 0.0, 0.0}, \
	{12, 3, 1.3, 1.8, 0.1, -0.1}, \
	{12, 4, -0.9, -2.2, -0.1, 0.0}, \
	{12, 5, 0.9, 0.3, 0.0, 0.0}, \
	{12, 6, 0.1, 0.7, 0.1, 0.0}, \
	{12, 7, 0.5, -0.1, 0.0, 0.0}, \
	{12, 8, -0.4, 0.3, 0.0, 0.0}, \
	{12, 9, -0.4, 0.2, 0.0, 0.0}, \
	{12, 10, 0.2, -0.9, 0.0, 0.0}, \
	{12, 11, -0.9, -0.2, 0.0, 0.0}, \
	{12, 12, 0.0, 0.7, 0.0, 0.0}}

#endif /* PHYSICAL_CONSTANTS_H_ */

/**
 * @}
 * @}
 */
