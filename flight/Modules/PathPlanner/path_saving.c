/**
 ******************************************************************************
 * @file       path_saving.c
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
 * @brief      Functions for loading and saving paths
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup PathPlanner Path Planner Module
 * @brief Executes a series of waypoints
 * @{
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
#include "openpilot.h"
#include "pios_flashfs.h"
#include "waypoint.h"

extern uintptr_t pios_waypoints_settings_fs_id;

/* Note: this system uses the flashfs in a slightly different way  */
/* the flashfs saves entries with an object and instance id. in    */
/* this code the object id is used to indicate the path id and the */
/* instance id is the waypoint number                              */

/**
 * Save the in memory waypoints to the waypoint filesystem
 * @param[in] id The path id to save as
 */
int32_t pathplanner_save_path(uint32_t path_id)
{
	WaypointData waypoint;

	if (WaypointHandle() == 0)
		return -30; // leave room for flashfs error codes

	uint32_t num_waypoints = UAVObjGetNumInstances(WaypointHandle());
	int32_t  waypoint_size = UAVObjGetNumBytes(WaypointHandle());
	int32_t  erase_retval  = 0;
	int32_t  last_save_id  = -1;
	int32_t  retval        = 0;

	// Save all elements
	for (int32_t i = 0; i < num_waypoints && retval == 0; i++) {
		WaypointInstGet(i, &waypoint);

		// Stop saving when get to invalid waypoint.  Nothing after or including is valid
		if (waypoint.Mode == WAYPOINT_MODE_INVALID)
			break;

		retval = PIOS_FLASHFS_ObjSave(pios_waypoints_settings_fs_id, path_id, i, (uint8_t *) &waypoint, waypoint_size);
		last_save_id = i; // Track the last valid waypoint id
	}

	// Check for any waypoints after the saved end of the path and erase them
	for (int32_t i = last_save_id + 1; erase_retval == 0; i++) {
		erase_retval = PIOS_FLASHFS_ObjLoad(pios_waypoints_settings_fs_id, path_id, i, (uint8_t *) &waypoint, waypoint_size);
		if (erase_retval == 0) {
			PIOS_FLASHFS_ObjDelete(pios_waypoints_settings_fs_id, path_id, i);
		}
	}

	return retval;
}

/**
 * Load a path from the waypoint filesystem into memory
 * @param[in] id The path id to load
 * @return -30 waypoint object not registered
 * @return -31 could not allocate waypoint in ram
 * @return other indicates FlashFS error
 */
int32_t pathplanner_load_path(uint32_t path_id)
{
	WaypointData waypoint;

	if (WaypointHandle() == 0)
		return -30; // leave room for flashfs error codes

	int32_t  waypoint_size = UAVObjGetNumBytes(WaypointHandle());
	int32_t  retval = 0;

	int32_t i;

	for (i = 0; retval == 0; i++) {
		retval = PIOS_FLASHFS_ObjLoad(pios_waypoints_settings_fs_id, path_id, i, (uint8_t *) &waypoint, waypoint_size);
		if (retval == 0) {
			// Loaded waypoint locally, store in UAVO manager
			if (i >= UAVObjGetNumInstances(WaypointHandle())) {
				int32_t new_instance_id = WaypointCreateInstance();
				if (new_instance_id != i) {
					retval = -31;
					break;
				}
			}

			WaypointInstSet(i, &waypoint);
		}
	}

	// Set any remaining waypoints to INVALID to indicate they should not be used
	// at this point i will be the index of the first waypoint that could not be
	// loaded from flash.
	for (; i <  UAVObjGetNumInstances(WaypointHandle()); i++) {
		WaypointInstGet(i, &waypoint);
		waypoint.Mode = WAYPOINT_MODE_INVALID;
		WaypointInstSet(i, &waypoint);
	}

	// Indicates instance not found. Successfully found end of path.
	if (retval == -3)
		return 0;

	return -2;
}