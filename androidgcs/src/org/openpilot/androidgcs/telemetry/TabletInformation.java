package org.openpilot.androidgcs.telemetry;

import org.openpilot.uavtalk.UAVObject;
import org.openpilot.uavtalk.UAVObjectField;
import org.openpilot.uavtalk.UAVObjectManager;

import android.content.Context;
import android.location.Criteria;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.util.Log;

public class TabletInformation {
	@SuppressWarnings("unused")
	private final static String TAG = TabletInformation.class.getSimpleName();

	private final UAVObjectManager objMngr;
	private final LocationManager locationManager;

	public TabletInformation(UAVObjectManager objMngr, Context service) {

		this.objMngr = objMngr;

		locationManager = (LocationManager)service.getSystemService(Context.LOCATION_SERVICE);

		Criteria criteria = new Criteria();
		criteria.setAccuracy(Criteria.ACCURACY_FINE);
		criteria.setAltitudeRequired(false);
		criteria.setBearingRequired(false);
		criteria.setCostAllowed(true);
		criteria.setPowerRequirement(Criteria.POWER_LOW);

		String provider = locationManager.getBestProvider(criteria, true);

		locationManager.requestLocationUpdates(provider,
                1000, // 1 second
                100,   // 1km
                locationListener);
	}

	public void stop() {
		locationManager.removeUpdates(locationListener);
	}

	LocationListener locationListener = new LocationListener() {

		@Override
		public void onLocationChanged(Location location) {

			Log.d("TabletInformation", "Location changed");

			UAVObject obj = objMngr.getObject("TabletInfo");
			if (obj == null)
				return;

			UAVObjectField field = obj.getField("Latitude");
			if (field == null)
				return;
			field.setValue(location.getLatitude() * 10e6);

			field = obj.getField("Longitude");
			if (field == null)
				return;
			field.setValue(location.getLongitude() * 10e6);

			field = obj.getField("Altitude");
			if (field == null)
				return;
			if (location.hasAltitude())
				field.setValue(location.getAltitude());
			else
				field.setValue("NAN");

			obj.updated();
		}

		@Override
		public void onProviderDisabled(String provider) {
			// TODO Auto-generated method stub

		}

		@Override
		public void onProviderEnabled(String provider) {
			// TODO Auto-generated method stub

		}

		@Override
		public void onStatusChanged(String provider, int status, Bundle extras) {
			// TODO Auto-generated method stub

		}

	};

}
