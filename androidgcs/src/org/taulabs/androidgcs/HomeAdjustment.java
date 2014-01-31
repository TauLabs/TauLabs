/**
 ******************************************************************************
 * @file       AttitudeAdjustment.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Refine aspects of the state estimation.
 * @see        The GNU Public License (GPL) Version 3
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
package org.taulabs.androidgcs;

import org.taulabs.androidgcs.drawer.NavDrawerActivityConfiguration;
import org.taulabs.uavtalk.UAVObject;

import android.content.Context;
import android.content.SharedPreferences;
import android.location.Location;
import android.location.LocationManager;
import android.os.Bundle;
import android.view.View;


public class HomeAdjustment extends ObjectManagerActivity {

	final String TAG = HomeAdjustment.class.getSimpleName();

	final boolean VERBOSE = false;
	final boolean DEBUG = true;

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
	}

	@Override
	protected NavDrawerActivityConfiguration getNavDrawerConfiguration() {
		NavDrawerActivityConfiguration navDrawer = getDefaultNavDrawerConfiguration();
		navDrawer.setMainLayout(R.layout.home_adjustment);
		return navDrawer;
	}
	
	@Override
	public void objectUpdated(UAVObject obj) {
	}

	/**
	 * Verify disarmed and if so force HomeLocation to unset.  This
	 * triggers an update of the home location and should reset the
	 * state estimator when required.
	 * @param v The button pressed
	 */
	public void homeToUav(View v) {
		if (!disarmed())
			return;

		UAVObject home = objMngr.getObject("HomeLocation");
		if (home == null)
			return;

		// This will trigger the UAV to update the home location
		home.getField("Set").setValue("FALSE");
		home.updated();
	}

	//! Set the home location altitude to the current GPS altitude
	public void homeAltitudeToUav(View v) {
		if (!disarmed())
			return;

		UAVObject home = objMngr.getObject("HomeLocation");
		UAVObject gps = objMngr.getObject("GPSPosition");
		if (home == null || gps == null)
			return;

		// TODO: for an update of GPS and perform the calculation
		// based on new results

		// The GPS altitude is consistently corrected by the
		// geoid separation on the flight controller so must
		// be done when setting the home location
		double altitude = gps.getField("Altitude").getDouble() +
				gps.getField("GeoidSeparation").getDouble();
		home.getField("Altitude").setValue(altitude);
		home.updated();
	}

	public void homeToTablet(View v) {

	}

	/**
	 * Work out the offset required to set tablet POI location to the 
	 * current UAV location
	 * 
	 * The difference is computed between the current GPS Position and
	 * TabletInfo and stored as an offset.  This offset is applied to
	 * the GPS position before sending it to the UAV for all updates,
	 * which is what is used for the POI location.  The android location
	 * on the map will still show up as what it perceives it to be and 
	 * not with the offset.
	 */
	public void poiToUAV(View v) {
		UAVObject gps = objMngr.getObject("GPSPosition");
		UAVObject tablet = objMngr.getObject("TabletInfo");
		UAVObject home = objMngr.getObject("HomeLocation");
		
		if (gps == null || tablet == null || home == null)
			return;

		LocationManager locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
		Location current = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
		
		// Altitude on the flight controller is always used as altitude
		// plug geoid separation
		double gps_altitude = gps.getField("Altitude").getDouble() +
				gps.getField("GeoidSeparation").getDouble();
		double alt_offset = gps_altitude - current.getLatitude();
		
		double lat_offset = gps.getField("Latitude").getDouble() - current.getLatitude() * 10e6;
		double lon_offset = gps.getField("Longitude").getDouble() - current.getLongitude() * 10e6;
		
		// Store these settings
		SharedPreferences settings = getSharedPreferences("TabletOffset", 0);
		SharedPreferences.Editor editor = settings.edit();
		// These values are all scaled up by 1E7 and fit in an int
		editor.putInt("alt_offset", (int) alt_offset);
		editor.putInt("lat_offset", (int) lat_offset);
		editor.putInt("lon_offset", (int) lon_offset);
		editor.commit();		
		
	}

	//! Verify the FC id disarmed
	private boolean disarmed() {
		if (objMngr == null)
			return false;

		// TODO: Force an update of the flight status object

		// verify the UAV is not armed
		UAVObject flightStatus = objMngr.getObject("FlightStatus");
		if (flightStatus == null)
			return false;
		if (flightStatus.getField("Armed").getValue().toString().compareTo("Disarmed") != 0)
			return false;

		return true;
	}
}
