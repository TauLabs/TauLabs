/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup GeoFenceModule Geo-fence Module
 * @brief Measures geo-fence position
 * @{
 *
 * @file       geofence.c
 * @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
 * @brief      Module to monitor position with respect to geo-fence and set alarms appropriately.
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
#include "coordinate_conversions.h"
#include "physical_constants.h"

#include "modulesettings.h"
#include "geofencevertices.h"
#include "geofencefaces.h"
#include "gpsposition.h"
#include "homelocation.h"
#include "positionactual.h"
#include "velocityactual.h"

//
// Configuration
//
#define STACK_SIZE_BYTES   1500
#define SAMPLE_PERIOD_MS   500
#define TASK_PRIORITY      (tskIDLE_PRIORITY + 1)

// Private functions
static void geofenceTask(void *parameters);
static void set_geo_fence_error(SystemAlarmsGeoFenceOptions error_code);
static bool check_enabled();

//! Recompute the translation from LLA to NED
static void HomeLocationUpdatedCb(UAVObjEvent * objEv);

//! Convert LLA to NED
static int32_t LLA2NED(int32_t LL[2], float altitude, float *NED);

//! Calculate ray-triangle intersection
static bool intersect_triangle( const float V0[3], const float V1[3],const float V2[3], const float  O[3], const float  D[3], float *t);


// Private types

// Private variables
static bool geofence_enabled = false;
static float geoidSeparation;
static xTaskHandle geofenceTaskHandle;
static HomeLocationData homeLocation;

//! Predict 3 seconds into the future
static const float safety_buffer_time = 3; // TODO: should perhaps not be hardcoded


/**
 * Initialise the geofence module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t GeoFenceStart(void)
{
	if (geofence_enabled) {
		// Start geofence task
		xTaskCreate(geofenceTask, (signed char *)"GeoFence", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &geofenceTaskHandle);
		TaskMonitorAdd(TASKINFO_RUNNING_GEOFENCE, geofenceTaskHandle);
		return 0;
	}
	return -1;
}


/**
 * Initialise the geofence module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t GeoFenceInitialize(void)
{
	geofence_enabled = check_enabled();

	if (geofence_enabled) {
		// Initialize UAVOs
		GPSPositionInitialize();
		HomeLocationInitialize();
		GeoFenceFacesInitialize();
		GeoFenceVerticesInitialize();

		HomeLocationConnectCallback(&HomeLocationUpdatedCb);

		return 0;
	}
	
	return -1;
}

MODULE_INITCALL(GeoFenceInitialize, GeoFenceStart);

/**
 * Main geo-fence task. It does not return.
 */
static void geofenceTask(void *parameters)
{
	HomeLocationUpdatedCb((UAVObjEvent *)NULL);

	// Main task loop
	portTickType lastSysTime = xTaskGetTickCount();
	portTickType lastGeoidUpdateTime = lastSysTime + 1;
	while(1) {
		//Once every minute update the geoid separation
		if(((lastGeoidUpdateTime - lastSysTime) * SAMPLE_PERIOD_MS > 60000) || (lastSysTime < lastGeoidUpdateTime)) {
			GPSPositionGeoidSeparationGet(&geoidSeparation);
			lastGeoidUpdateTime = lastSysTime;
		}

		vTaskDelayUntil(&lastSysTime, SAMPLE_PERIOD_MS * portTICK_RATE_MS);

		uint8_t sum_crossings_buffer_zone = 0; //<-- This could just be a bool that is toggled each time there's a crossing
		uint8_t sum_crossings_safe_zone = 0; //<-- This could just be a bool that is toggled each time there's a crossing
		
		uint16_t num_vertices=UAVObjGetNumInstances(GeoFenceVerticesHandle());
		uint16_t num_faces=UAVObjGetNumInstances(GeoFenceFacesHandle());

		if (num_vertices < 4) {// The fewest number of vertices requiered to make a 3D volume is 4.
			set_geo_fence_error(SYSTEMALARMS_GEOFENCE_INSUFFICIENTVERTICES);
			continue;
		}
		if (num_faces < 4) {// The fewest number of faces requiered to make a 3D volume is 4.
			set_geo_fence_error(SYSTEMALARMS_GEOFENCE_INSUFFICIENTFACES);
			continue;
		}

		VelocityActualData velocityActualData;
		PositionActualData positionActual;

		//Load UAVOs
		PositionActualGet(&positionActual);
		VelocityActualGet(&velocityActualData);

		// Ray origin
		float O[3] ={positionActual.North,  positionActual.East,  positionActual.Down};

		// Ray direction
		float D[3] = {velocityActualData.North, velocityActualData.East, velocityActualData.Down};
		// Handle the case where the vehicle is stopped, and thus there is no directionality to the velocity vector
		if (D[0] == 0 && D[1] == 0 && D[2] == 0)
			D[0] = 1;


		// Loop through all faces, testing for intersection between ray and triangle
		for (typeof(num_faces) i=0; i<num_faces; i++) {
			GeoFenceVerticesData geofenceVerticesData;
			GeoFenceFacesData geofenceFacesData;
			
			//Get the face of interest
			GeoFenceFacesInstGet(i, &geofenceFacesData);
			
			//Get the three face vertices and convert into NED. Vertex order is important!
			GeoFenceVerticesInstGet(geofenceFacesData.Vertices[0], &geofenceVerticesData);
			int32_t vertexA_LLA[2]={geofenceVerticesData.Latitude, geofenceVerticesData.Longitude};
			float vertexA[3];
			LLA2NED(vertexA_LLA, geofenceVerticesData.Altitude, vertexA);

			GeoFenceVerticesInstGet(geofenceFacesData.Vertices[1], &geofenceVerticesData);
			int32_t vertexB_LLA[2]={geofenceVerticesData.Latitude, geofenceVerticesData.Longitude};
			float vertexB[3];
			LLA2NED(vertexB_LLA, geofenceVerticesData.Altitude, vertexB);

			GeoFenceVerticesInstGet(geofenceFacesData.Vertices[2], &geofenceVerticesData);
			int32_t vertexC_LLA[2]={geofenceVerticesData.Latitude, geofenceVerticesData.Longitude};
			float vertexC[3];
			LLA2NED(vertexC_LLA, geofenceVerticesData.Altitude, vertexC);

			float t_now = -1;

			// Test if ray falls inside triangle. Since the ray's direction is the velocity vector, then
			// the returned value, t, is exactly equal to the time to intersection
			bool inside = intersect_triangle(vertexA, vertexB, vertexC, O, D, &t_now);

			// Check ray results
			if (inside == false) // If no intersection, then continue
				continue;
			else if (t_now < 0) // If no positive intersection, then continue
				continue;
			else if (t_now < safety_buffer_time) { // The vehicle is in the safety buffer zone
				sum_crossings_buffer_zone++;
			}
			else { // The vehicle is safely inside the geo-fence
				sum_crossings_safe_zone++;
				sum_crossings_buffer_zone++;
			}
		}

		//Tests if we have crossed the geo-fence
		if (sum_crossings_safe_zone % 2) {	//If there are an odd number of faces crossed, then the UAV is and will be inside the polyhedron.
			set_geo_fence_error(SYSTEMALARMS_GEOFENCE_NONE);
		}
		else if (sum_crossings_buffer_zone % 2) {	//If sum_crossings_safe_zone is even but sum_crossings_buffer_zone is odd, then the UAV is inside the polyhedron but is leaving soon.
			set_geo_fence_error(SYSTEMALARMS_GEOFENCE_LEAVINGBOUNDARY);
		}
		else { //If sum_crossings_buffer_zone is even, then the UAV is outside the polyhedron.
			set_geo_fence_error(SYSTEMALARMS_GEOFENCE_LEFTBOUNDARY);
		}
		
	}
	
}


/**
 * @brief triangle_intersection "Fast Minimum Storage Ray Triange Intersection", Moller
 * and Trumbore, 1997. Calculates the intersection between a line parameterized by R(t) = O + tD,
 * and a triangle defined by vertices V0, V1, and V2
 * @param[in] V0 Triangle vertex A
 * @param[in] V1 Triangle vertex B
 * @param[in] V2 Triangle vertex C
 * @param[in] O 3D Ray origin
 * @param[in] D 3D Ray direction
 * @param[out] t intersection line parameter
 * @return returns true if the line intersects the triangle, false if not
 */
static bool intersect_triangle(const float V0[3], const float V1[3], const float V2[3], const float  O[3], const float  D[3], float *t)
{
	// Find vectors for two edges sharing V0
	float edge1[3];
	float edge2[3];
	for (int i=0; i<3; i++) {
		edge1[i] = V1[i] - V0[i];
		edge2[i] = V2[i] - V0[i];
	}

	// Begin calculating determinant - also used to calculate u parameter
	float P[3];
	CrossProduct(D, edge2, P); // P = D x e2

	// If determinant is near zero, ray lies in plane of triangle
#define EPSILON .000001f
	float det = DotProduct(edge1, P);
	if(det > -EPSILON && det < EPSILON)
		return false;

	// Calculate distance from V0 to ray origin
	float T[3];
	for (int i=0; i<3; i++)
		T[i] = O[i] - V0[i];

	// Calculate u parameter and test bound
	float inv_det = 1.0f / det;
	float u = DotProduct(T, P) * inv_det;

	// The intersection lies outside of the triangle
	if(u < 0.0f || u > 1.0f)
		return false;

	// Prepare to test v parameter
	float Q[3];
	CrossProduct(T, edge1, Q); // Q = T x e1;

	// Calculate V parameter and test bound
	float v = DotProduct(D, Q) * inv_det;

	// The intersection lies outside of the triangle
	if(v < 0.0f || u + v  > 1.0f)
		return false;

	// Compute line parameter for R(t) = O + tD
	*t = DotProduct(edge2, Q) * inv_det;
	return true;
}


/**
* @brief check_enabled Checks the proper configuration of all requisite modules and variables
* @return true if Geo-fencing can be enabled, false otherwise
*/
static bool check_enabled()
{
	ModuleSettingsInitialize();
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];

	ModuleSettingsAdminStateGet(module_state);

	bool homelocation_set = false;      //Geo-fence only works if the home location is set
	bool gps_module_enabled = false;    //Geo-fence only works if we have GPS or groundtruth
	bool groundtruth_available = false; //Geo-fence only works if we have GPS or groundtruth
	bool pathfollower_module_enabled = false; // Geo-fence only works if we can autonomously steer the vehicle
	bool geofence_module_enabled = false;

	HomeLocationGet(&homeLocation);
	if (homeLocation.Set == HOMELOCATION_SET_TRUE)
		homelocation_set = true;

#ifdef MODULE_GPS_BUILTIN
	gps_module_enabled = true;
#else
	if (module_state[MODULESETTINGS_ADMINSTATE_GPS] == MODULESETTINGS_ADMINSTATE_ENABLED)
		gps_module_enabled = true;
#endif

#ifdef MODULE_PATHFOLLOWER_BUILTIN
	pathfollower_module_enabled = true;
#else
if (module_state[MODULESETTINGS_ADMINSTATE_VTOLPATHFOLLOWER] == MODULESETTINGS_ADMINSTATE_ENABLED ||
		module_state[MODULESETTINGS_ADMINSTATE_FIXEDWINGPATHFOLLOWER] == MODULESETTINGS_ADMINSTATE_ENABLED ||
		module_state[MODULESETTINGS_ADMINSTATE_GROUNDPATHFOLLOWER] == MODULESETTINGS_ADMINSTATE_ENABLED) {
	pathfollower_module_enabled=true;
}
#endif

#ifdef MODULE_GEOFENCE_BUILTIN
	geofence_module_enabled = true;
#else
	if (module_state[MODULESETTINGS_ADMINSTATE_GEOFENCE] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		geofence_module_enabled=true;
	}
#endif

	return (gps_module_enabled || groundtruth_available) && pathfollower_module_enabled && geofence_module_enabled && homelocation_set;
}


/**
 * @brief set_geo_fence_error Set the error code and alarm state
 * @param[in] error_code
 */
static void set_geo_fence_error(SystemAlarmsGeoFenceOptions error_code)
{
	// Get the severity of the alarm given the error code
	SystemAlarmsAlarmOptions severity;
	switch (error_code) {
		case SYSTEMALARMS_GEOFENCE_NONE:
		severity = SYSTEMALARMS_ALARM_OK;
		break;
	case SYSTEMALARMS_GEOFENCE_LEAVINGBOUNDARY:
		severity = SYSTEMALARMS_ALARM_WARNING;
		break;
	case SYSTEMALARMS_GEOFENCE_LEFTBOUNDARY:
		severity = SYSTEMALARMS_ALARM_CRITICAL;
		break;
	case SYSTEMALARMS_GEOFENCE_INSUFFICIENTVERTICES:
		severity = SYSTEMALARMS_ALARM_ERROR;
		break;
	case SYSTEMALARMS_GEOFENCE_INSUFFICIENTFACES:
		severity = SYSTEMALARMS_ALARM_ERROR;
		break;
	default:
		severity = SYSTEMALARMS_ALARM_ERROR;
		error_code = SYSTEMALARMS_CONFIGERROR_UNDEFINED;
		break;
	}

	// Make sure not to set the error code if it didn't change
	SystemAlarmsGeoFenceOptions current_error_code;
	SystemAlarmsGeoFenceGet((uint8_t *) &current_error_code);
	if (current_error_code != error_code) {
		SystemAlarmsGeoFenceSet((uint8_t *) &error_code);
	}

	// AlarmSet checks only updates on toggle
	AlarmsSet(SYSTEMALARMS_ALARM_GEOFENCE, (uint8_t) severity);
}


/**
 * @brief Convert the Lat-Lon-Altitude position into NED coordinates
 * @note this method uses a taylor expansion around the home coordinates
 * to convert to NED which allows it to be done with only floating point
 * calculations
 * @param[in] World frame coordinates, (Lat, Lon, Altitude)
 * @param[out] NED frame coordinates
 * @returns 0 for success, -1 for failure
 */
static float T[3];
static int32_t LLA2NED(int32_t LL[2], float altitude, float *NED)
{
	float dL[3] = { (LL[0] - homeLocation.Latitude) / 10.0e6f * DEG2RAD,
		(LL[1] - homeLocation.Longitude) / 10.0e6f * DEG2RAD,
		(altitude + geoidSeparation - homeLocation.Altitude)
	};

	NED[0] = T[0] * dL[0];
	NED[1] = T[1] * dL[1];
	NED[2] = T[2] * dL[2];

	return 0;
}


/**
 * @brief HomeLocationUpdatedCb Recompute the translation from LLA to NED
 * @param objEv
 */
static void HomeLocationUpdatedCb(UAVObjEvent *objEv)
{
	float lat;
	float alt;

	HomeLocationGet(&homeLocation);

	// Compute vector for converting deltaLLA to NED
	lat = homeLocation.Latitude / 10.0e6f * DEG2RAD;
	alt = homeLocation.Altitude;

	T[0] = alt + 6.378137E6f;
	T[1] = cosf(lat) * (alt + 6.378137E6f);
	T[2] = -1.0f;
}


/**
 * @}
 */
