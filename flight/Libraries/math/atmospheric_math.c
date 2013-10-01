/**
 ******************************************************************************
 * @file       atmospheric_math.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup Math Utilities
 * @{
 * @addtogroup MiscellaneousMath Math Various mathematical routines
 * @{
 * @brief Miscellaneous math support
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

#include "math.h"

#include "atmospheric_math.h" 		/* API declarations */
#include "physical_constants.h"

/**
 * @brief air_density_from_altitude calculate air density from altitude. http://en.wikipedia.org/wiki/Density_of_air
 * @param alt
 * @param air
 * @param gravity
 * @return
 */
float air_density_from_altitude(float altitude, struct AirParameters *air)
{
	float pressure = air_pressure_from_altitude(altitude, air);
	float rho = pressure*air->M / (air->univ_gas_constant*(air->air_temperature_at_surface - air->temperature_lapse_rate*altitude));

	return rho;
}

/**
 * @brief air_pressure_from_altitude Get air pressure from altitude and atmospheric conditions.
 * @param altitude altitude
 * @param air atmospheric conditions
 * @param gravity
 * @return
 */
float air_pressure_from_altitude(float altitude, struct AirParameters *air)
{
	float pressure = air->sea_level_press* powf(1 - air->temperature_lapse_rate*altitude / air->air_temperature_at_surface, GRAVITY*air->M / (air->univ_gas_constant*air->temperature_lapse_rate));

	return pressure;
}

/**
 * @brief cas2tas calculate TAS from CAS and altitude. http://en.wikipedia.org/wiki/Airspeed
 * @param CAS Calibrated airspeed
 * @param altitude altitude
 * @param air atmospheric conditions
 * @param gravity
 * @return TAS True airspeed
 */
float cas2tas(float CAS, float altitude, struct AirParameters *air)
{
	float rho=air_density_from_altitude(altitude, air);
	float TAS = CAS * sqrtf(air->air_density_at_surface/rho);

	return TAS;
}

/**
 * @brief tas2cas calculate CAS from TAS and altitude. http://en.wikipedia.org/wiki/Airspeed
 * @param TAS True airspeed
 * @param altitude altitude
 * @param air atmospheric conditions
 * @param gravity
 * @return CAS Calibrated airspeed
 */
float tas2cas(float TAS, float altitude, struct AirParameters *air)
{
	float rho=air_density_from_altitude(altitude, air);
	float CAS = TAS / sqrtf(air->air_density_at_surface/rho);

	return CAS;
}


/**
 * @brief initialize_air_structure Initializes the structure with standard-temperature-pressure values
 * @return
 */
struct AirParameters initialize_air_structure()
{
	struct AirParameters air;

	air.sea_level_press = STANDARD_AIR_SEA_LEVEL_PRESSURE;
	air.air_density_at_surface = STANDARD_AIR_DENSITY;
	air.air_temperature_at_surface = STANDARD_AIR_TEMPERATURE;
	air.temperature_lapse_rate = STANDARD_AIR_LAPSE_RATE;
	air.univ_gas_constant = UNIVERSAL_GAS_CONSTANT;
	air.dry_air_constant = DRY_AIR_CONSTANT;
	air.relative_humidity = STANDARD_AIR_RELATIVE_HUMIDITY;
	air.M = STANDARD_AIR_MOLS2KG;

	return air;
}
