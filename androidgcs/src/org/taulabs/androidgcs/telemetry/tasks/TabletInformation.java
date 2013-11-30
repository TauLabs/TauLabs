/**
 ******************************************************************************
 * @file       TabletInformation.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      A telemetry task which constantly updates the UAV with the
 *             tablet location
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
package org.taulabs.androidgcs.telemetry.tasks;

import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectField;
import org.taulabs.uavtalk.UAVObjectManager;

import android.content.Context;
import android.content.SharedPreferences;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.util.Log;

public class TabletInformation {
	private final static String TAG = TabletInformation.class.getSimpleName();
	private final static boolean DEBUG = false;

	private UAVObjectManager objMngr;
	private LocationManager locationManager;
	private Context mService;

	public TabletInformation() {}

	public void connect(UAVObjectManager objMngr, Context service) {

		this.objMngr = objMngr;
		this.mService = service;

		locationManager = (LocationManager)service.getSystemService(Context.LOCATION_SERVICE);

		if (DEBUG) Log.d(TAG, "Connecting to location updates");
		locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER,
                1000, // 1 second
                0,    // 1 m
                locationListener);
	}

	public void disconnect() {
		if (locationManager != null)
			locationManager.removeUpdates(locationListener);
	}

	LocationListener locationListener = new LocationListener() {

		@Override
		public void onLocationChanged(Location location) {

			if (DEBUG) Log.d("TabletInformation", "Location changed");

			UAVObject obj = objMngr.getObject("TabletInfo");
			if (obj == null)
				return;
			SharedPreferences settings = mService.getSharedPreferences("TabletOffset", 0);
			int alt_offset = settings.getInt("alt_offset", 0);
			int lat_offset = settings.getInt("lat_offset", 0);
			int lon_offset = settings.getInt("lon_offset", 0);
			
			UAVObjectField field = obj.getField("Latitude");
			if (field == null)
				return;
			field.setValue(location.getLatitude() * 10e6 + lat_offset);

			field = obj.getField("Longitude");
			if (field == null)
				return;
			field.setValue(location.getLongitude() * 10e6 + lon_offset);

			field = obj.getField("Altitude");
			if (field == null)
				return;
			if (location.hasAltitude())
				field.setValue(alt_offset + location.getAltitude());
			else
				field.setValue("0");

			field = obj.getField("Connected");
			if (field != null) {
				field.setValue("True");
			}

			obj.updated();
		}

		@Override
		public void onProviderDisabled(String provider) {
			if (DEBUG) Log.d(TAG, "Provider disabled");
		}

		@Override
		public void onProviderEnabled(String provider) {
			if (DEBUG) Log.d(TAG, "Provider enabled");
		}

		@Override
		public void onStatusChanged(String provider, int status, Bundle extras) {
			if (DEBUG) Log.d(TAG, "Status changed");
		}

	};

}
