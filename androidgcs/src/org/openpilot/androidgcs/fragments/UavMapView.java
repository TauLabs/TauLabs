package org.openpilot.androidgcs.fragments;

import java.util.ArrayList;

import org.openpilot.androidgcs.R;
import org.openpilot.uavtalk.UAVObject;
import org.openpilot.uavtalk.UAVObjectField;
import org.openpilot.uavtalk.UAVObjectManager;
import org.osmdroid.ResourceProxy;
import org.osmdroid.api.IGeoPoint;
import org.osmdroid.tileprovider.tilesource.XYTileSource;
import org.osmdroid.util.GeoPoint;
import org.osmdroid.views.MapView;
import org.osmdroid.views.overlay.ItemizedOverlay;
import org.osmdroid.views.overlay.MyLocationOverlay;
import org.osmdroid.views.overlay.Overlay;
import org.osmdroid.views.overlay.OverlayItem;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Point;
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

	//! The overlay which display path desired
	private PathDesiredOverlay pathDesiredOverlay;
	//! The overlay which display the UAV symbol and Home
	private UavLocationOverlay uavLocationOverlay;

	//! Cache the home location
	private GeoPoint homeLocation;
	//! Cache the uav location
	private GeoPoint uavLocation;
	//! Cache the path desired
	private GeoPoint pathDesired;

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
		XYTileSource tileSource = new XYTileSource("myTMStilesource", null, 3, 20, 256, ".png", "http://oatile1.mqcdn.com/tiles/1.0.0/sat/");
		mOsmv.setTileSource(tileSource);
		mOsmv.setBuiltInZoomControls(true);
		mOsmv.setMultiTouchControls(true);
		mOsmv.getController().setZoom(15);

		// Some useful commands for other data sources
		//XYTileSource tileSource = new XYTileSource("myTMStilesource", null, 3, 20, 256, ".png", "http://wms.jpl.nasa.gov/wms.cgi?");
		//mOsmv.setTileSource(TileSourceFactory.MAPNIK);
		//mOsmv.setUseDataConnection(false);

		// Add the overlay which shows home and the UAV
		uavLocationOverlay = new UavLocationOverlay(getActivity());
		mOsmv.getOverlays().add(uavLocationOverlay);

		// Add an overlay that shows path navigation and the position desired
		pathDesiredOverlay = new PathDesiredOverlay(getActivity());
		mOsmv.getOverlays().add(pathDesiredOverlay);

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

		obj = objMngr.getObject("PathDesired");
		if (obj != null)
			registerObjectUpdates(obj);
		objectUpdated(obj);

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

		if (obj.getName().compareTo("PathDesired") == 0) {
			pathDesired = getPathDesiredLocation();
		}

        mOsmv.invalidate();
	}

	/**
	 * Convert from an NED representation to a geopoint
	 * @param[in] NED the NED location, altitude not used
	 * @return GeoPoint for this location
	 */
	private GeoPoint convertLocation(double NED[]) {
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

		// Compute the LLA coordinates
		lat = lat + (NED[0] / T0) * 180.0 / Math.PI;
		lon = lon + (NED[1] / T1) * 180.0 / Math.PI;

		return new GeoPoint((int) (lat * 1e6), (int) (lon * 1e6));
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

		// Get the NED coordinates
		double NED[] = new double[2];
		NED[0] = pos.getField("North").getDouble();
		NED[1] = pos.getField("East").getDouble();

		return convertLocation(NED);
	}

	/**
	 * Convert the UAV location in NED representation to an
	 * longitude latitude GeoPoint
	 * @return The location as a GeoPoint
	 */
	private GeoPoint getPathDesiredLocation() {
		UAVObject pos = objMngr.getObject("PathDesired");
		if (pos == null)
			return new GeoPoint(0,0);

		// Get the NED coordinates
		double NED[] = new double[2];
		NED[0] = pos.getField("End").getDouble(0);
		NED[1] = pos.getField("End").getDouble(1);

		return convertLocation(NED);
	}

	//! An overlay that shows the path desired location
	class PathDesiredOverlay extends Overlay
	{
		private final Drawable waypointMarker;
	    public PathDesiredOverlay(Context ctx) {
			super(ctx);
			waypointMarker = getResources().getDrawable(R.drawable.marker_default);
		}

		@Override
		protected void draw(Canvas canvas, MapView arg1, boolean arg2) {
			Point screenPoint = new Point();
			mOsmv.getProjection().toMapPixels(pathDesired, screenPoint);

			waypointMarker.setBounds(screenPoint.x-waypointMarker.getIntrinsicWidth() / 2,
					screenPoint.y-waypointMarker.getIntrinsicHeight() / 2,
					screenPoint.x+waypointMarker.getIntrinsicWidth() / 2,
					screenPoint.y+waypointMarker.getIntrinsicHeight() / 2);
			waypointMarker.draw(canvas);
		}
	}

	//! An overlay that shows the path desired location
	class UavLocationOverlay extends Overlay
	{
		private final Drawable homeMarker;
		private final Drawable uavMarker;

	    public UavLocationOverlay(Context ctx) {
			super(ctx);
			homeMarker = getResources().getDrawable(R.drawable.ic_home);
			uavMarker  = getResources().getDrawable(R.drawable.ic_uav);
		}

		@Override
		protected void draw(Canvas canvas, MapView arg1, boolean arg2) {
			Point screenPoint = new Point();
			mOsmv.getProjection().toMapPixels(homeLocation, screenPoint);

			homeMarker.setBounds(screenPoint.x-homeMarker.getIntrinsicWidth() / 2, screenPoint.y-homeMarker.getIntrinsicHeight()/2,
					screenPoint.x+homeMarker.getIntrinsicWidth() / 2, screenPoint.y+homeMarker.getIntrinsicHeight() / 2);
			homeMarker.draw(canvas);

			screenPoint = new Point();
			mOsmv.getProjection().toMapPixels(uavLocation, screenPoint);

			final int UAV_SIZE = 64;
			uavMarker.setBounds(screenPoint.x-UAV_SIZE/2, screenPoint.y-UAV_SIZE/2,
					screenPoint.x+UAV_SIZE/2 , screenPoint.y+UAV_SIZE/2);
			uavMarker.draw(canvas);
		}
	}

}
