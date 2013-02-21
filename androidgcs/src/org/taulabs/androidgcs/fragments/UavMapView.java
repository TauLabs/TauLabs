/**
 ******************************************************************************
 * @file       UavMapView.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      A map view that uses OSMDroid (and caches data for off-line work)
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
package org.taulabs.androidgcs.fragments;

import java.util.ArrayList;
import java.util.List;

import org.taulabs.androidgcs.R;
import org.osmdroid.ResourceProxy;
import org.osmdroid.api.IGeoPoint;
import org.osmdroid.api.IMapView;
import org.osmdroid.tileprovider.tilesource.XYTileSource;
import org.osmdroid.util.GeoPoint;
import org.osmdroid.views.MapView;
import org.osmdroid.views.MapView.Projection;
import org.osmdroid.views.overlay.ItemizedIconOverlay;
import org.osmdroid.views.overlay.ItemizedOverlay;
import org.osmdroid.views.overlay.MyLocationOverlay;
import org.osmdroid.views.overlay.Overlay;
import org.osmdroid.views.overlay.OverlayItem;
import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectField;
import org.taulabs.uavtalk.UAVObjectManager;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Point;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.RelativeLayout;

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
	private WaypointsOverlay pathDesiredOverlay;
	//! The overlay which display the UAV symbol and Home
	private UavLocationOverlay uavLocationOverlay;

	//! Cache the home location
	private GeoPoint homeLocation;
	//! Cache the uav location
	private GeoPoint uavLocation;
	//! Cache the path desired
	private GeoPoint pathDesired;
	//! Cache the heading
	private float yaw;

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
		pathDesiredOverlay = new WaypointsOverlay(getResources().getDrawable(R.drawable.marker_default),mResourceProxy); //new PathDesiredOverlay(mResourceProxy);
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

		obj = objMngr.getObject("AttitudeActual");
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
			pathDesiredOverlay.update();
		}

		if (obj.getName().compareTo("AttitudeActual") == 0) {
			UAVObjectField field = obj.getField("Yaw");
			if (field != null)
				yaw = (float) field.getDouble();
			return;
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

	private double[] getNED(GeoPoint coord) {
		double [] NED = new double[3];

		UAVObject home = objMngr.getObject("HomeLocation");
		if (home == null)
			return NED;

		double lat = home.getField("Latitude").getDouble() / 10.0e6;
		double lon = home.getField("Longitude").getDouble() / 10.0e6;
		double alt = home.getField("Altitude").getDouble();

	    double T[] = new double[3];
	    T[0] = alt+6.378137E6f * Math.PI / 180.0;
	    T[1] = Math.cos(lat * Math.PI / 180.0)*(alt+6.378137E6f) * Math.PI / 180.0;
	    T[2] = -1.0f;

	    NED[0] = (coord.getLatitudeE6() / 1e6 - lat) * T[0];
	    NED[1] = (coord.getLongitudeE6() / 1e6 - lon) * T[1];
	    NED[2] = (coord.getAltitude() - alt) * T[2];

		return NED;
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

	protected class PathGesture implements ItemizedIconOverlay.OnItemGestureListener<OverlayItem> {

		@Override
		public boolean onItemLongPress(int i, OverlayItem item) {
			Log.d(TAG, "LongPress: " + i + " " + item);
			return false;
		}

		@Override
		public boolean onItemSingleTapUp(int i, OverlayItem item) {
			Log.d(TAG, "SingleTap: " + i + " " + item);
			return false;
		}

	};

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
			if (homeLocation == null || uavLocation == null)
				return;

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
			canvas.save();
			canvas.rotate(yaw,screenPoint.x,screenPoint.y);
			uavMarker.draw(canvas);
			canvas.restore();
		}
	}

	private class WaypointsOverlay extends ItemizedOverlay<OverlayItem> {
		private final List<OverlayItem> items=new ArrayList<OverlayItem>();
		private Drawable marker=null;
		private OverlayItem inDrag=null;
		private ImageView dragImage=null;
		private int xDragImageOffset=0;
		private int yDragImageOffset=0;
		private int xDragTouchOffset=0;
		private int yDragTouchOffset=0;

		public WaypointsOverlay(Drawable marker, ResourceProxy proxy) {
			super(marker,proxy);
			this.marker=marker;

			dragImage=(ImageView)getView().findViewById(R.id.drag_waypoint);
			xDragImageOffset=dragImage.getDrawable().getIntrinsicWidth()/2;
			yDragImageOffset=dragImage.getDrawable().getIntrinsicHeight();

			update();
			populate();
		}

		public void update() {
			items.clear();
			if (pathDesired != null)
				items.add(new OverlayItem("Waypoint", "PathDesired", pathDesired));
			populate();
		}

		@Override
		protected OverlayItem createItem(int i) {
			return(items.get(i));
		}

		@Override
		public void draw(Canvas canvas, MapView mapView,
				boolean shadow) {
			super.draw(canvas, mapView, shadow);

			//boundCenterBottom(marker);
		}

		@Override
		public int size() {
			return(items.size());
		}

		@Override
		public boolean onTouchEvent(MotionEvent event, MapView mapView) {
			Log.d(TAG, "onTouchEvent");
			final int action=event.getAction();
			final int x=(int)event.getX();
			final int y=(int)event.getY();
			boolean result=false;

			if (action==MotionEvent.ACTION_DOWN) {
				for (OverlayItem item : items) {
					Point p=new Point(0,0);
					//GeoPoint mapCenter = (GeoPoint) mOsmv.getMapCenter();

					//mOsmv.getProjection().toMapPixels(item.getPoint(), p);
					p = pointFromGeoPoint(item.getPoint(), mOsmv);
					if (p == null)
						return false;

					//Log.d(TAG, "ACTION_DOWN: (" + x + " " + p.x + ", " + y + " " + p.y + ")");

					if (hitTest(item, marker, x-p.x, y-p.y)) {
						Log.d(TAG, "Found hit");
						result=true;
						inDrag=item;
						items.remove(inDrag);
						populate();

						xDragTouchOffset=0;
						yDragTouchOffset=0;

						setDragImagePosition(p.x, p.y);
						dragImage.setVisibility(View.VISIBLE);

						xDragTouchOffset=x-p.x;
						yDragTouchOffset=y-p.y;

						break;
					}
				}
			}
			else if (action==MotionEvent.ACTION_MOVE && inDrag!=null) {
				setDragImagePosition(x, y);
				result=true;
			}
			else if (action==MotionEvent.ACTION_UP && inDrag!=null) {
				Log.d(TAG, "Found drop");
				dragImage.setVisibility(View.GONE);

				GeoPoint pt=(GeoPoint) mOsmv.getProjection().fromPixels(x-xDragTouchOffset,
						y-yDragTouchOffset);
				OverlayItem toDrop=new OverlayItem(inDrag.getSnippet(), inDrag.getTitle(),
						pt);

				items.add(toDrop);
				populate();

				inDrag=null;
				result=true;

				double NED[] = getNED(pt);
				UAVObject obj = objMngr.getObject("PathDesired");
				obj.getField("End").setValue(NED[0], 0);
				obj.getField("End").setValue(NED[1], 1);
				obj.updated();
			}

			return(result || super.onTouchEvent(event, mapView));
		}

		private void setDragImagePosition(int x, int y) {
			RelativeLayout.LayoutParams lp=
					(RelativeLayout.LayoutParams)dragImage.getLayoutParams();

			lp.setMargins(x-xDragImageOffset-xDragTouchOffset,
					y-yDragImageOffset-yDragTouchOffset, 0, 0);
			dragImage.setLayoutParams(lp);
		}

		@Override
		public boolean onSnapToItem(int arg0, int arg1, Point arg2,
				IMapView arg3) {
			return false;
		}
	}

	/**
	 *
	 * @param gp GeoPoint
	 * @param vw Mapview
	 * @return a 'Point' in screen coords relative to top left
	 */
	private Point pointFromGeoPoint(GeoPoint gp, MapView vw){

	    Point rtnPoint = new Point();
	    Projection projection = vw.getProjection();
	    projection.toPixels(gp, rtnPoint);
	    // Get the top left GeoPoint
	    GeoPoint geoPointTopLeft = (GeoPoint) projection.fromPixels(0, 0);
	    Point topLeftPoint = new Point();
	    // Get the top left Point (includes osmdroid offsets)
	    projection.toPixels(geoPointTopLeft, topLeftPoint);
	    rtnPoint.x-= topLeftPoint.x; // remove offsets
	    rtnPoint.y-= topLeftPoint.y;
	    if (rtnPoint.x > vw.getWidth() || rtnPoint.y > vw.getHeight() ||
	            rtnPoint.x < 0 || rtnPoint.y < 0){
	        return null; // gp must be off the screen
	    }
	    return rtnPoint;
	}

}
