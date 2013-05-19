/**
 ******************************************************************************
 *
 * @file       example_program.c
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

#include "openpilot.h"
#include "uavobjectmanager.h"

#include "pathsegmentdescriptor.h"
#include "positionactual.h"
#include "fixedwingairspeeds.h"

// private functions

void example_program()
{
	PathSegmentDescriptorData pathSegmentDescriptor;
	
	for (int i=UAVObjGetNumInstances(PathSegmentDescriptorHandle()); i<=6; i++){
		PathSegmentDescriptorCreateInstance();
	}
	
	PositionActualData positionActual;
	PositionActualGet(&positionActual);

	FixedWingAirspeedsData fixedWingAirspeeds;
	FixedWingAirspeedsGet(&fixedWingAirspeeds);

	// First locus is current vehicle position
	pathSegmentDescriptor.SwitchingLocus[0] = positionActual.North;
	pathSegmentDescriptor.SwitchingLocus[1] = positionActual.East;
	pathSegmentDescriptor.SwitchingLocus[2] = positionActual.Down;
	pathSegmentDescriptor.FinalVelocity = fixedWingAirspeeds.BestClimbRateSpeed;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.NumberOfOrbits = 0;
	pathSegmentDescriptor.PathCurvature = 0;
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(0, &pathSegmentDescriptor);

	pathSegmentDescriptor.SwitchingLocus[0] = -100;
	pathSegmentDescriptor.SwitchingLocus[1] = -100;
	pathSegmentDescriptor.SwitchingLocus[2] = -50;
	pathSegmentDescriptor.FinalVelocity = fixedWingAirspeeds.BestClimbRateSpeed;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.PathCurvature = 0;
	pathSegmentDescriptor.NumberOfOrbits = 0;
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(1, &pathSegmentDescriptor);

	pathSegmentDescriptor.SwitchingLocus[0] = 100;
	pathSegmentDescriptor.SwitchingLocus[1] = 0;
	pathSegmentDescriptor.SwitchingLocus[2] = -50;
	pathSegmentDescriptor.FinalVelocity = fixedWingAirspeeds.BestClimbRateSpeed;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.PathCurvature = 0;
	pathSegmentDescriptor.NumberOfOrbits = 0;
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(2, &pathSegmentDescriptor);

	pathSegmentDescriptor.SwitchingLocus[0] = 100;
	pathSegmentDescriptor.SwitchingLocus[1] = 120;
	pathSegmentDescriptor.SwitchingLocus[2] = -50;
	pathSegmentDescriptor.FinalVelocity = fixedWingAirspeeds.BestClimbRateSpeed;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.PathCurvature = 1/70.0f; // 70m radius
	pathSegmentDescriptor.NumberOfOrbits = 1;
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(3, &pathSegmentDescriptor);

	pathSegmentDescriptor.SwitchingLocus[0] = 75;
	pathSegmentDescriptor.SwitchingLocus[1] = 100;
	pathSegmentDescriptor.SwitchingLocus[2] = -50;
	pathSegmentDescriptor.FinalVelocity = fixedWingAirspeeds.BestClimbRateSpeed;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.PathCurvature = 0;
	pathSegmentDescriptor.NumberOfOrbits = 0;
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(4, &pathSegmentDescriptor);
	
	pathSegmentDescriptor.SwitchingLocus[0] = 75;
	pathSegmentDescriptor.SwitchingLocus[1] = 25;
	pathSegmentDescriptor.SwitchingLocus[2] = -50;
	pathSegmentDescriptor.FinalVelocity = fixedWingAirspeeds.BestClimbRateSpeed;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.PathCurvature = 1/400.0f;
	pathSegmentDescriptor.NumberOfOrbits = 0;
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(5, &pathSegmentDescriptor);

	// Circle home location
	pathSegmentDescriptor.SwitchingLocus[0] = 0;
	pathSegmentDescriptor.SwitchingLocus[1] = 0;
	pathSegmentDescriptor.SwitchingLocus[2] = -50;
	pathSegmentDescriptor.FinalVelocity = fixedWingAirspeeds.BestClimbRateSpeed;
	pathSegmentDescriptor.DesiredAcceleration = 0;
	pathSegmentDescriptor.PathCurvature = 1/60.0f; //60m radius
	pathSegmentDescriptor.NumberOfOrbits = 1e8; //TODO: Define this really large floating-point value as a magic number
	pathSegmentDescriptor.ArcRank = PATHSEGMENTDESCRIPTOR_ARCRANK_MINOR;
	PathSegmentDescriptorInstSet(6, &pathSegmentDescriptor);
}
