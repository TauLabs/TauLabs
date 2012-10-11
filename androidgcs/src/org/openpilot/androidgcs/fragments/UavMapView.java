package org.openpilot.androidgcs.fragments;

import java.util.ArrayList;

import org.openpilot.androidgcs.R;
import org.openpilot.uavtalk.UAVObject;
import org.openpilot.uavtalk.UAVObjectManager;
import org.osmdroid.ResourceProxy;
import org.osmdroid.api.IGeoPoint;
import org.osmdroid.util.GeoPoint;
import org.osmdroid.views.MapView;
import org.osmdroid.views.overlay.ItemizedIconOverlay;
import org.osmdroid.views.overlay.ItemizedOverlay;
import org.osmdroid.views.overlay.MyLocationOverlay;
import org.osmdroid.views.overlay.OverlayItem;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

public class UavMapView extends ObjectManagerFragment {

	private static final String TAG = MapView.class.getSimpleName();
	private static final int LOGLEVEL = 0;
	// private static boolean WARN = LOGLEVEL > 1;
	private static final boolean DEBUG = LOGLEVEL > 0;

	protected MapView mOsmv;
	protected MyLocationOverlay mLocationOverlay;
	protected ItemizedOverlay<OverlayItem> mUavOverlay;
	protected ResourceProxy mResourceProxy;
	public IGeoPoint currentLocation;

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

		if(mLocationOverlay != null) {
			mOsmv.getOverlays().add(this.mLocationOverlay);
			mLocationOverlay.enableMyLocation();
			mLocationOverlay.enableFollowLocation();
		} else {
			Log.e(TAG, "Unable to create map overlay");
		}

		// Create overlay for the UAV and Home
        ArrayList<OverlayItem> items = new ArrayList<OverlayItem>();
        // Put overlay icon a little way from map center
        GeoPoint uavLocation = new GeoPoint(29.7631*1e6, -95.3631*1e6);
        GeoPoint homeLocation = new GeoPoint(29.7651*1e6, -95.3631*1e6);
        items.add(new OverlayItem("UAV", "The current UAV location", uavLocation));
        items.add(new OverlayItem("Home", "The home location", homeLocation));

        mUavOverlay = new ItemizedIconOverlay<OverlayItem>(items,
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
        	mOsmv.getOverlays().add(this.mUavOverlay);
        }
	}

	@Override
	public void onOPConnected(UAVObjectManager objMngr) {
		super.onOPConnected(objMngr);
		if (DEBUG)
			Log.d(TAG, "On connected");

		UAVObject obj;

		obj = objMngr.getObject("PositionActual");
		if (obj != null)
			registerObjectUpdates(obj);
		objectUpdated(obj);

		obj = objMngr.getObject("HomeLocation");
		if (obj != null)
			registerObjectUpdates(obj);
		objectUpdated(obj);
	}

	/**
	 * Called whenever any objects subscribed to via registerObjects
	 */
	@Override
	public void objectUpdated(UAVObject obj) {
		if (DEBUG)
			Log.d(TAG, "Updated");

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
