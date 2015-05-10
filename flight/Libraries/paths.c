/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       paths.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @brief      Path calculation library with common API
 * 
 * Paths are represented by the structure @ref PathDesired and also take in
 * @ref PositionActual.  This library then computes the error from the path
 * which includes the vector tangent to the path at the closest location
 * and the distance of that vector.  The distance along the path is also
 * returned in the path_status.
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
#include "paths.h"

#include "uavobjectmanager.h"
#include "pathdesired.h"

// private functions
static void path_endpoint(const float * start_point, const float * end_point, 
                          const float * cur_point, struct path_status * status);
static void path_vector(const float * start_point, const float * end_point, 
                        const float * cur_point, struct path_status * status);
static void path_circle(const float * center_point, const float radius, 
                        const float * cur_point, struct path_status * status,
                        bool clockwise);
static void path_curve(const float * start_point, const float * end_point,
                       const float radius, const float * cur_point, 
                       struct path_status * status, bool clockwise);

/**
 * @brief Compute progress along path and deviation from it
 * @param[in] start_point Starting point
 * @param[in] end_point Ending point
 * @param[in] cur_point Current location
 * @param[in] mode Path following mode
 * @param[out] status Structure containing progress along path and deviation
 */
void path_progress(const PathDesiredData *pathDesired,
                   const float *cur_point,
                   struct path_status *status)
{
	uint8_t mode = pathDesired->Mode;
	float start_point[2] = {pathDesired->Start[0],pathDesired->Start[1]};
	float end_point[2] = {pathDesired->End[0],pathDesired->End[1]};

	switch(mode) {
		case PATHDESIRED_MODE_FLYVECTOR:
		case PATHDESIRED_MODE_DRIVEVECTOR:
			return path_vector(start_point, end_point, cur_point, status);
			break;
		case PATHDESIRED_MODE_FLYCIRCLERIGHT:
		case PATHDESIRED_MODE_DRIVECIRCLERIGHT:
			return path_curve(start_point, end_point, pathDesired->ModeParameters, cur_point, status, 1);
			break;
		case PATHDESIRED_MODE_FLYCIRCLELEFT:
		case PATHDESIRED_MODE_DRIVECIRCLELEFT:
			return path_curve(start_point, end_point, pathDesired->ModeParameters, cur_point, status, 0);
			break;
		case PATHDESIRED_MODE_CIRCLEPOSITIONLEFT:
			return path_circle(end_point, pathDesired->ModeParameters, cur_point, status, 0);
			break;
		case PATHDESIRED_MODE_CIRCLEPOSITIONRIGHT:
			return path_circle(end_point, pathDesired->ModeParameters, cur_point, status, 1);
			break;
		case PATHDESIRED_MODE_FLYENDPOINT:
		case PATHDESIRED_MODE_DRIVEENDPOINT:
		case PATHDESIRED_MODE_HOLDPOSITION:
		default:
			// use the endpoint as default failsafe if called in unknown modes
			return path_endpoint(start_point, end_point, cur_point, status);
			break;
	}
}

/**
 * @brief Compute progress towards endpoint. Deviation equals distance
 * @param[in] start_point Starting point
 * @param[in] end_point Ending point
 * @param[in] cur_point Current location
 * @param[out] status Structure containing progress along path and deviation
 */
static void path_endpoint(const float *start_point,
                          const float *end_point,
                          const float *cur_point,
                          struct path_status *status)
{
	float path_north, path_east, diff_north, diff_east;
	float dist_path, dist_diff;

	// we do not correct in this mode
	status->correction_direction[0] = status->correction_direction[1] = 0;

	// Distance to go
	path_north = end_point[0] - start_point[0];
	path_east = end_point[1] - start_point[1];

	// Current progress location relative to end
	diff_north = end_point[0] - cur_point[0];
	diff_east = end_point[1] - cur_point[1];

	dist_diff = sqrtf( diff_north * diff_north + diff_east * diff_east );
	dist_path = sqrtf( path_north * path_north + path_east * path_east );

	if(dist_diff < 1e-6f ) {
		status->fractional_progress = 1;
		status->error = 0;
		status->path_direction[0] = status->path_direction[1] = 0;
		return;
	}

	status->fractional_progress = 1 - dist_diff / (1 + dist_path);
	status->error = dist_diff;

	// Compute direction to travel
	status->path_direction[0] = diff_north / dist_diff;
	status->path_direction[1] = diff_east / dist_diff;

}

/**
 * @brief Compute progress along path and deviation from it
 * @param[in] start_point Starting point
 * @param[in] end_point Ending point
 * @param[in] cur_point Current location
 * @param[out] status Structure containing progress along path and deviation
 */
static void path_vector(const float *start_point,
                        const float *end_point,
                        const float *cur_point,
                        struct path_status *status)
{
	float path_north, path_east, diff_north, diff_east;
	float dist_path;
	float dot;
	float normal[2];

	// Distance to go
	path_north = end_point[0] - start_point[0];
	path_east = end_point[1] - start_point[1];

	// Current progress location relative to start
	diff_north = cur_point[0] - start_point[0];
	diff_east = cur_point[1] - start_point[1];

	dot = path_north * diff_north + path_east * diff_east;
	dist_path = sqrtf( path_north * path_north + path_east * path_east );

	if(dist_path < 1e-6f) {
		// if the path is too short, we cannot determine vector direction.
		// Fly towards the endpoint to prevent flying away,
		// but assume progress=1 either way.
		path_endpoint( start_point, end_point, cur_point, status );
		status->fractional_progress = 1;
		return;
	}

	// Compute the normal to the path
	normal[0] = -path_east / dist_path;
	normal[1] = path_north / dist_path;

	status->fractional_progress = dot / (dist_path * dist_path);
	status->error = normal[0] * diff_north + normal[1] * diff_east;

	// Compute direction to correct error
	status->correction_direction[0] = (status->error > 0) ? -normal[0] : normal[0];
	status->correction_direction[1] = (status->error > 0) ? -normal[1] : normal[1];
	
	// Now just want magnitude of error
	status->error = fabs(status->error);

	// Compute direction to travel
	status->path_direction[0] = path_north / dist_path;
	status->path_direction[1] = path_east / dist_path;

}

/**
 * @brief Circle location continuously
 * @param[in] start_point Starting point
 * @param[in] end_point Center point
 * @param[in] cur_point Current location
 * @param[out] status Structure containing progress along path and deviation
 */
static void path_circle(const float * center_point,
                        const float radius,
                        const float * cur_point,
                        struct path_status * status, 
                        bool clockwise)
{
	float diff_north, diff_east;
	float cradius;
	float normal[2];

	// Current location relative to center
	diff_north = cur_point[0] - center_point[0];
	diff_east = cur_point[1] - center_point[1];

	cradius = sqrtf(  diff_north * diff_north   +   diff_east * diff_east );

	if (cradius < 1e-6f) {
		// cradius is zero, just fly somewhere and make sure correction is still a normal
		status->fractional_progress = 1;
		status->error = radius;
		status->correction_direction[0] = 0;
		status->correction_direction[1] = 1;
		status->path_direction[0] = 1;
		status->path_direction[1] = 0;
		return;
	}

	if (clockwise) {
		// Compute the normal to the radius clockwise
		normal[0] = -diff_east / cradius;
		normal[1] = diff_north / cradius;
	} else {
		// Compute the normal to the radius counter clockwise
		normal[0] = diff_east / cradius;
		normal[1] = -diff_north / cradius;
	}
	
	status->fractional_progress = 0;

	// error is current radius minus wanted radius - positive if too close
	status->error = radius - cradius;

	// Compute direction to correct error
	status->correction_direction[0] = (status->error>0?1:-1) * diff_north / cradius;
	status->correction_direction[1] = (status->error>0?1:-1) * diff_east / cradius;

	// Compute direction to travel
	status->path_direction[0] = normal[0];
	status->path_direction[1] = normal[1];

	status->error = fabs(status->error);
}

/**
 * @brief Compute progress along circular path and deviation from it
 * @param[in] start_point Starting point
 * @param[in] end_point Ending point
 * @param[in] radius Radius of the curve segment
 * @param[in] cur_point Current location
 * @param[out] status Structure containing progress along path and deviation
 */
static void path_curve(const float * start_point,
                       const float * end_point,
                       const float radius,
                       const float * cur_point,
                       struct path_status *status,
                       bool clockwise)
{
	float diff_north, diff_east;
	float path_north, path_east;
	float cradius;
	float normal[2];	

	// Compute the center of the circle connecting the two points as the intersection of two circles
	// around the two points from
	// http://www.mathworks.com/matlabcentral/newsreader/view_thread/255121
	float m_n, m_e, p_n, p_e, d, center[2];

	// Center between start and end
	m_n = (start_point[0] + end_point[0]) / 2;
	m_e = (start_point[1] + end_point[1]) / 2;

	// Normal vector the line between start and end.
	if (clockwise) {
		p_n = -(end_point[1] - start_point[1]);
		p_e = (end_point[0] - start_point[0]);
	} else {
		p_n = (end_point[1] - start_point[1]);
		p_e = -(end_point[0] - start_point[0]);		
	}

	// Work out how far to go along the perpendicular bisector
	d = sqrtf(radius * radius / (p_n * p_n + p_e * p_e) - 0.25f);

	float radius_sign = (radius > 0) ? 1 : -1;
	float m_radius = fabs(radius);

	if (fabs(p_n) < 1e-3 && fabs(p_e) < 1e-3) {
		center[0] = m_n;
		center[1] = m_e;
	} else {
		center[0] = m_n + p_n * d * radius_sign;
		center[1] = m_e + p_e * d * radius_sign;
	}

	// Current location relative to center
	diff_north = cur_point[0] - center[0];
	diff_east = cur_point[1] - center[1];

	// Compute current radius from the center
	cradius = sqrtf(  diff_north * diff_north   +   diff_east * diff_east );

	// Compute error in terms of meters from the curve (the distance projected
	// normal onto the path i.e. cross-track distance)
	status->error = m_radius - cradius;

	if (cradius < 1e-6f) {
		// cradius is zero, just fly somewhere and make sure correction is still a normal
		status->fractional_progress = 1;
		status->error = m_radius;
		status->correction_direction[0] = 0;
		status->correction_direction[1] = 1;
		status->path_direction[0] = 1;
		status->path_direction[1] = 0;
		return;
	}

	if (clockwise) {
		// Compute the normal to the radius clockwise
		normal[0] = -diff_east / cradius;
		normal[1] = diff_north / cradius;
	} else {
		// Compute the normal to the radius counter clockwise
		normal[0] = diff_east / cradius;
		normal[1] = -diff_north / cradius;
	}

	// Compute direction to correct error
	status->correction_direction[0] = (status->error>0?1:-1) * diff_north / cradius;
	status->correction_direction[1] = (status->error>0?1:-1) * diff_east / cradius;

	// Compute direction to travel
	status->path_direction[0] = normal[0];
	status->path_direction[1] = normal[1];

	path_north = end_point[0] - start_point[0];
	path_east = end_point[1] - start_point[1];
	diff_north = cur_point[0] - start_point[0];
	diff_east = cur_point[1] - start_point[1];
	float dist_path = sqrtf( path_north * path_north + path_east * path_east );
	float dot = path_north * diff_north + path_east * diff_east;

	status->fractional_progress = dot / (dist_path * dist_path);

	status->error = fabs(status->error);
}

/**
 * @}
 */
