package org.openpilot.androidgcs.fragments;

import java.util.ArrayList;

import org.openpilot.androidgcs.R;
import org.openpilot.uavtalk.UAVObject;
import org.openpilot.uavtalk.UAVObjectField;
import org.openpilot.uavtalk.UAVObjectManager;
import org.osmdroid.ResourceProxy;
import org.osmdroid.api.IGeoPoint;
import org.osmdroid.util.GeoPoint;
import org.osmdroid.views.MapView;
import org.osmdroid.views.overlay.ItemizedIconOverlay;
import org.osmdroid.views.overlay.ItemizedOverlay;
import org.osmdroid.views.overlay.MyLocationOverlay;
import org.osmdroid.views.overlay.OverlayItem;

import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

public class UavMapView extends ObjectManagerFragment {

	private static final String TAG = MapView.class.getSimpleName();
	private static final int LOGLEVEL = 0;
	// private static boolean WARN = LOGLEVEL > 1;
	private static final boolean DEBUG = LOGLEVEL > 2;
	private static final boolean ERROR = LOGLEVEL > 0;

	protected MapView mOsmv;
	protected MyLocationOverlay mLocationOverlay;
	protected ItemizedOverlay<OverlayItem> mUavOverlay;
	protected ResourceProxy mResourceProxy;
	public IGeoPoint currentLocation;
	ArrayList<OverlayItem> mItems;

	//! Cache the home location
	private GeoPoint homeLocation;
	//! Cache the uav location
	private GeoPoint uavLocation;

	// @Override
	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
			Bundle savedInstanceState) {
		// Inflate the layout for this fragment
		Log.d(TAG, "Expanding the map view");

		return inflater.inflate(R.layout.map, container, false);
	}

	@Override
	public void onResume() {
		super.onResume();

		mResourceProxy = new ResourceProxyImpl(getActivity());

		mOsmv = (MapView) getActivity().findViewById(R.id.mapview);
		//Assert.assertNotNull(mOsmv);

		mLocationOverlay = new MyLocationOverlay(getActivity(), mOsmv, mResourceProxy);
		mOsmv.setBuiltInZoomControls(true);
		mOsmv.setMultiTouchControls(true);
		mOsmv.getController().setZoom(13);

		mItems = new ArrayList<OverlayItem>();

		if(mLocationOverlay != null) {
			mOsmv.getOverlays().add(this.mLocationOverlay);
			mLocationOverlay.enableMyLocation();
			mLocationOverlay.enableFollowLocation();
		} else {
			if (ERROR) Log.e(TAG, "Unable to create map overlay");
		}

	}

	@Override
	public void onOPConnected(UAVObjectManager objMngr) {
		super.onOPConnected(objMngr);
		if (DEBUG) Log.d(TAG, "On connected");

		UAVObject obj;

		obj = objMngr.getObject("PositionActual");
		if (obj != null)
			registerObjectUpdates(obj);
		objectUpdated(obj);

		obj = objMngr.getObject("HomeLocation");
		obj.updateRequested(); // Make sure this is correct
		if (obj != null)
			registerObjectUpdates(obj);
		objectUpdated(obj);
	}

	/**
	 * Called whenever any objects subscribed to via registerObjects
	 */
	@Override
	public void objectUpdated(UAVObject obj) {
		if (DEBUG) Log.d(TAG, "Updated");

		if (obj.getName().compareTo("HomeLocation") == 0) {
			double lat = 0, lon = 0;

			UAVObject home = objMngr.getObject("HomeLocation");
			if (home != null) {
				UAVObjectField latField = home.getField("Latitude");
				if (latField != null)
					lat = latField.getDouble() / 10e6;

				UAVObjectField lonField = home.getField("Longitude");
				if (lonField != null)
					lon = lonField.getDouble() / 10e6;
			}

			homeLocation = new GeoPoint(lat, lon);
		}
		if (obj.getName().compareTo("PositionActual") == 0) {
			uavLocation = getUavLocation();
		}

		// Create items for home and uav with nice icons
		OverlayItem uav = new OverlayItem("UAV", "The current UAV location", uavLocation);
		Drawable icon = getResources().getDrawable(R.drawable.ic_uav);
		icon.setBounds(0, 0, icon.getIntrinsicWidth(), icon.getIntrinsicHeight());
		uav.setMarker(icon);

		OverlayItem home = new OverlayItem("Home", "The home location", homeLocation);
		icon = getResources().getDrawable(R.drawable.ic_home);
		icon.setBounds(0, 0, icon.getIntrinsicWidth(), icon.getIntrinsicHeight());
		home.setMarker(icon);

        mItems.clear();
        mItems.add(uav);
        mItems.add(home);

        synchronized(mOsmv) {
        	if (mOsmv.getOverlays().size() > 1)
        		mOsmv.getOverlays().remove( 1 );

        	mUavOverlay = new ItemizedIconOverlay<OverlayItem>(mItems,
        			new ItemizedIconOverlay.OnItemGestureListener<OverlayItem>() {
        		@Override
        		public boolean onItemSingleTapUp(final int index,
        				final OverlayItem item) {
        			return true; // We 'handled' this event.
        		}
        		@Override
        		public boolean onItemLongPress(final int index,
        				final OverlayItem item) {
        			return false;
        		}
        	}, mResourceProxy);
        	if (mUavOverlay != null) {
        		mOsmv.getOverlays().add(mUavOverlay);
        	}
        }

        mOsmv.invalidate();
	}

	/**
	 * Convert the UAV location in NED representation to an
	 * longitude latitude GeoPoint
	 * @return The location as a GeoPoint
	 */
	private GeoPoint getUavLocation() {
		UAVObject pos = objMngr.getObject("PositionActual");
		if (pos == null)
			return new GeoPoint(0,0);

		UAVObject home = objMngr.getObject("HomeLocation");
		if (home == null)
			return new GeoPoint(0,0);

		double lat, lon, alt;
		lat = home.getField("Latitude").getDouble() / 10.0e6;
		lon = home.getField("Longitude").getDouble() / 10.0e6;
		alt = home.getField("Altitude").getDouble();

		// Get the home coordinates
		double T0, T1;
		T0 = alt+6.378137E6;
		T1 = Math.cos(lat * Math.PI / 180.0)*(alt+6.378137E6);

		// Get the NED coordinates
		double NED0, NED1;
		NED0 = pos.getField("North").getDouble();
		NED1 = pos.getField("East").getDouble();

		// Compute the LLA coordinates
		lat = lat + (NED0 / T0) * 180.0 / Math.PI;
		lon = lon + (NED1 / T1) * 180.0 / Math.PI;

		return new GeoPoint((int) (lat * 1e6), (int) (lon * 1e6));
	}

	/*
	public class MyLocationListener implements LocationListener {

	    public void onLocationChanged(Location location) {
	        currentLocation = new GeoPoint(location);
	        displayMyCurrentLocationOverlay();
	    }

	    public void onProviderDisabled(String provider) {
	    }

	    public void onProviderEnabled(String provider) {
	    }

	    public void onStatusChanged(String provider, int status, Bundle extras) {
	    }
	}

	@Override
	protected void onCreate(Bundle savedInstanceState) {
	    super.onCreate(savedInstanceState);
	    locationListener = new MyLocationListener();
	    locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
	    locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0, 0, locationListener);
	    Location location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
	    if( location != null ) {
	        currentLocation = new GeoPoint(location.getLatitude(), location.getLongitude());
	    }
	}

	private void displayMyCurrentLocationOverlay() {
	    if( currentLocation != null) {
	        if( currentLocationOverlay == null ) {
	            currentLocationOverlay = new ArrayItemizedOverlay(myLocationMarker);
	            myCurrentLocationOverlayItem = new OverlayItem(currentLocation, "My Location", "My Location!");
	            currentLocationOverlay.addItem(myCurrentLocationOverlayItem);
	            mOsmv.getOverlays().add(currentLocationOverlay);
	        } else {
	            myCurrentLocationOverlayItem.setPoint(currentLocation);
	            currentLocationOverlay.requestRedraw();
	        }
	        mOsmv.getController().setCenter(currentLocation);
	    }
	}
	*/
}
