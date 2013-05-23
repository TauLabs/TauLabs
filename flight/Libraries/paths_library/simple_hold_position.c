/**
 ******************************************************************************
 *
 * @file       simple_return_to_home.c
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
 * @brief      Library path manipulation 
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

#include "pios.h"

#include "uavobjectmanager.h"

#include "pathsegmentdescriptor.h"
#include "positionactual.h"
#include "fixedwingairspeeds.h"

// private functions

void simple_hold_position()
{
	PathSegmentDescriptorData pathSegmentDescriptor;
	
	for (int i=UAVObjGetNumInstances(PathSegmentDescriptorHandle()); i<=1; i++){
		PathSegmentDescriptorCreateInstance();
	}
	
	PositionActualData positionActual;
	PositionActualGet(&positionActual);

    FixedWingAirspeedsData fixedWingAirspeeds;
    FixedWingAirspeedsGet(&fixedWingAirspeeds);

	float radius = 60;
	
	// First locus is current vehicle position
	pathSegmentDescriptor.SwitchingLocus[0] = positionActual.North - radius;
	pathSegmentDescriptor.SwitchingLocus[1] = positionActual.East;
	pathSegmentDescriptor.SwitchingLocus[2] = positionActual.Down;
	pathSegmentDescriptor.FinalVelocity = fixedWingAirspeeds.BestClimbRateSpeed;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.NumberOfOrbits = 0;
	pathSegmentDescriptor.PathCurvature = 0;
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(0, &pathSegmentDescriptor);

	// Orbit current position
	pathSegmentDescriptor.SwitchingLocus[0] = positionActual.North + radius;
	pathSegmentDescriptor.SwitchingLocus[1] = positionActual.East;
	pathSegmentDescriptor.SwitchingLocus[2] = positionActual.Down - 10;
	pathSegmentDescriptor.FinalVelocity = fixedWingAirspeeds.BestClimbRateSpeed;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.PathCurvature = 1/radius;
	pathSegmentDescriptor.NumberOfOrbits = 1e8; //TODO: Define this really large floating-point value as a magic number
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(1, &pathSegmentDescriptor);
}
