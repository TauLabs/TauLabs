package org.taulabs.androidgcs.fragments;

import java.util.ArrayList;
import java.util.List;

import org.taulabs.androidgcs.R;
import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectField;
import org.taulabs.uavtalk.UAVObjectManager;

import android.content.Context;
import android.content.SharedPreferences;
import android.location.Location;
import android.location.LocationManager;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.util.Log;
import android.view.ContextMenu;
import android.view.LayoutInflater;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.ContextMenu.ContextMenuInfo;
import android.widget.ToggleButton;

import com.google.android.gms.common.GooglePlayServicesNotAvailableException;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.MapView;
import com.google.android.gms.maps.MapsInitializer;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.Marker;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.maps.GeoPoint;

public class Map extends ObjectManagerFragment {
	private final String TAG = Map.class.getSimpleName();
	private static int LOGLEVEL = 1;
	private static boolean DEBUG = LOGLEVEL > 0;

	private MapView m;
	private GoogleMap mMap;
	private Marker mUavMarker;
	private Marker mHomeMarker;
	private Marker mPoiMarker;
	private Marker mPathDesiredMarker;
	private final List<Marker> mWaypointMarkers = new ArrayList<Marker>();

    private GeoPoint homeLocation;
    private GeoPoint pathDesiredLocation;
    private GeoPoint uavLocation;
    private GeoPoint poiLocation;

	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
			Bundle savedInstanceState) {
		// inflate and return the layout
		View v = inflater.inflate(R.layout.map_fragment, container, false);
		m = (MapView) v.findViewById(R.id.mapView);
		m.onCreate(savedInstanceState);

		try {
		     MapsInitializer.initialize(getActivity());
		 } catch (GooglePlayServicesNotAvailableException e) {
		     e.printStackTrace();
		 }

		mMap = m.getMap();

		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(getActivity());
		int map_type = Integer.decode(prefs.getString("map_type", "1"));

		if (DEBUG) Log.d(TAG, "Map type selected: " + map_type);

        switch(map_type) {
        case 0:
        	mMap.setMapType(GoogleMap.MAP_TYPE_NORMAL);
        	break;
        case 1:
        	mMap.setMapType(GoogleMap.MAP_TYPE_SATELLITE);
        	break;
        case 2:
        	mMap.setMapType(GoogleMap.MAP_TYPE_TERRAIN);
        	break;
        case 3:
        	mMap.setMapType(GoogleMap.MAP_TYPE_HYBRID);
        	break;
        }
		mMap.setMyLocationEnabled(true);

		// Pull up context menu on a long click
		registerForContextMenu(m);
		mMap.setOnMapLongClickListener(new GoogleMap.OnMapLongClickListener() {
			@Override
			public void onMapLongClick(LatLng arg0) {
				Log.d(TAG, "Click");
				m.showContextMenu();
				// Animating to the touched position
                mMap.animateCamera(CameraUpdateFactory.newLatLng(arg0));
			}
		});
		
		mMap.setOnMarkerDragListener(dragListener);

		//! If the current tablet location is available, jump straight to it
		LocationManager locationManager =
		        (LocationManager) getActivity().getSystemService(Context.LOCATION_SERVICE);
		Location tabletLocation = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
		if (tabletLocation != null) {
			LatLng lla = new LatLng(tabletLocation.getLatitude(), tabletLocation.getLongitude());
			if (DEBUG) Log.d(TAG, "Location: " + lla);
			mMap.moveCamera(CameraUpdateFactory.newLatLngZoom(lla, 15));
		} else {
			if (DEBUG) Log.d(TAG, "Could not get location");
		}
		return v;
	}


	@Override
	public void onOPConnected(UAVObjectManager objMngr) {
		super.onOPConnected(objMngr);
		UAVObject obj = objMngr.getObject("HomeLocation");
		if (obj != null) {
			obj.updateRequested(); // Make sure this is correct and been updated
			registerObjectUpdates(obj);
			objectUpdated(obj);
		}

		obj = objMngr.getObject("PathDesired");
		if (obj != null) {
			obj.updateRequested(); // Make sure this is correct and been updated
			registerObjectUpdates(obj);
			objectUpdated(obj);
		}

		obj = objMngr.getObject("PoiLocation");
		if (obj != null) {
			obj.updateRequested(); // Make sure this is correct and been updated
			registerObjectUpdates(obj);
			objectUpdated(obj);
		}

		obj = objMngr.getObject("PositionActual");
		if (obj != null) {
			obj.updateRequested(); // Make sure this is correct and been updated
			registerObjectUpdates(obj);
			objectUpdated(obj);
		}

		obj = objMngr.getObject("TabletInfo");
		if (obj != null) {
			obj.updateRequested(); // Make sure this is correct and been updated
			registerObjectUpdates(obj);
			objectUpdated(obj);
		}

		obj = objMngr.getObject("Waypoint");
		if (obj != null) {
			obj.updateRequested(); // Make sure this is correct and been updated
			registerObjectUpdates(obj);
			objectUpdated(obj);
		}

		obj = objMngr.getObject("WaypointActive");
		if (obj != null) {
			obj.updateRequested(); // Make sure this is correct and been updated
			registerObjectUpdates(obj);
			objectUpdated(obj);
		}
	}
	
	//! Convert position actual into a GeoPoint
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

	//! Convert the POI location into a GeoPoint
	private GeoPoint getPoiLocation() {
		UAVObject poi = objMngr.getObject("PoiLocation");
		if (poi == null)
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
		NED0 = poi.getField("North").getDouble();
		NED1 = poi.getField("East").getDouble();

		// Compute the LLA coordinates
		lat = lat + (NED0 / T0) * 180.0 / Math.PI;
		lon = lon + (NED1 / T1) * 180.0 / Math.PI;

		return new GeoPoint((int) (lat * 1e6), (int) (lon * 1e6));
	}
	//! Convert path desired into a GeoPoint
	private GeoPoint getPathDesiredLocation() {
		UAVObject pos = objMngr.getObject("PathDesired");
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
		NED0 = pos.getField("End").getDouble(0);
		NED1 = pos.getField("End").getDouble(1);

		// Compute the LLA coordinates
		lat = lat + (NED0 / T0) * 180.0 / Math.PI;
		lon = lon + (NED1 / T1) * 180.0 / Math.PI;

		return new GeoPoint((int) (lat * 1e6), (int) (lon * 1e6));
	}

	//! Get a GeoPoint for a Waypoint
	private GeoPoint getWaypointLocation(int idx) {
		UAVObject pos = objMngr.getObject("Waypoint", idx);
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
		NED0 = pos.getField("Position").getDouble(0);
		NED1 = pos.getField("Position").getDouble(1);

		// Compute the LLA coordinates
		lat = lat + (NED0 / T0) * 180.0 / Math.PI;
		lon = lon + (NED1 / T1) * 180.0 / Math.PI;

		return new GeoPoint((int) (lat * 1e6), (int) (lon * 1e6));
	}
	/**
	 * Called whenever any objects subscribed to via registerObjects
	 * update the marker location for home and the UAV
	 */
	@Override
	public void objectUpdated(UAVObject obj) {
		if (obj == null)
			return;
		if (obj.getName().compareTo("HomeLocation") == 0) {
			Double lat = obj.getField("Latitude").getDouble() / 10;
			Double lon = obj.getField("Longitude").getDouble() / 10;
			homeLocation = new GeoPoint(lat.intValue(), lon.intValue());
			if (mHomeMarker == null) {
				mHomeMarker = mMap.addMarker(new MarkerOptions().anchor(0.5f,0.5f)
			       .position(new LatLng(homeLocation.getLatitudeE6() / 1e6, homeLocation.getLongitudeE6() / 1e6))
			       .title("Home")
			       .snippet(String.format("%g, %g", homeLocation.getLatitudeE6() / 1e6, homeLocation.getLongitudeE6() / 1e6))
			       .icon(BitmapDescriptorFactory.fromResource(R.drawable.im_map_home)));
			} else {
				mHomeMarker.setPosition((new LatLng(homeLocation.getLatitudeE6() / 1e6, homeLocation.getLongitudeE6() / 1e6)));
			}
		} else if (obj.getName().compareTo("TabletInfo") == 0) {
			UAVObjectField field = obj.getField("Connected");
			if (field == null || field.getValue().toString().compareTo("False") == 0)
				setTabletTracking(false);
			

		} else if (obj.getName().compareTo("PathDesired") == 0) {
			pathDesiredLocation = getPathDesiredLocation();
			if (mPathDesiredMarker == null) {
				mPathDesiredMarker = mMap.addMarker(new MarkerOptions().anchor(0.5f,1.0f)
			       .position(new LatLng(pathDesiredLocation.getLatitudeE6() / 1e6, pathDesiredLocation.getLongitudeE6() / 1e6))
			       .title("Path Desired")
			       .snippet(String.format("%g, %g", pathDesiredLocation.getLatitudeE6() / 1e6, pathDesiredLocation.getLongitudeE6() / 1e6))
			       .icon(BitmapDescriptorFactory.fromResource(R.drawable.marker_default)));
			} else {
				mPathDesiredMarker.setPosition((new LatLng(pathDesiredLocation.getLatitudeE6() / 1e6, pathDesiredLocation.getLongitudeE6() / 1e6)));
			}
			mPathDesiredMarker.setDraggable(true);

		} else if (obj.getName().compareTo("PoiLocation") == 0) {
			poiLocation = getPoiLocation();
			if (mPoiMarker == null) {
				mPoiMarker = mMap.addMarker(new MarkerOptions().anchor(0.5f,0.5f)
			       .position(new LatLng(poiLocation.getLatitudeE6() / 1e6, poiLocation.getLongitudeE6() / 1e6))
			       .title("POI")
			       .snippet(String.format("%g, %g", poiLocation.getLatitudeE6() / 1e6, poiLocation.getLongitudeE6() / 1e6))
			       .icon(BitmapDescriptorFactory.fromResource(R.drawable.im_map_poi)));
			} else {
				mPoiMarker.setPosition((new LatLng(poiLocation.getLatitudeE6() / 1e6, poiLocation.getLongitudeE6() / 1e6)));
			}
		} else if (obj.getName().compareTo("PositionActual") == 0) {
			uavLocation = getUavLocation();
			if (mUavMarker == null) {
				mUavMarker = mMap.addMarker(new MarkerOptions().anchor(0.5f,0.5f)
			       .position(new LatLng(uavLocation.getLatitudeE6() / 1e6, uavLocation.getLongitudeE6() / 1e6))
			       .title("UAV")
			       .snippet(String.format("%g, %g", uavLocation.getLatitudeE6() / 1e6, uavLocation.getLongitudeE6() / 1e6))
			       .icon(BitmapDescriptorFactory.fromResource(R.drawable.im_map_uav)));
			} else {
				mUavMarker.setPosition((new LatLng(uavLocation.getLatitudeE6() / 1e6, uavLocation.getLongitudeE6() / 1e6)));
			}
		} else if (obj.getName().compareTo("Waypoint") == 0) {
			int instances = objMngr.getNumInstances(obj.getObjID());
			for (int idx = 0; idx < instances; idx++) {
				GeoPoint pos = getWaypointLocation(idx);
				if (idx >= mWaypointMarkers.size()) {
					Marker waypointMarker = mMap.addMarker(new MarkerOptions().anchor(0.5f,1.0f)
							.position(new LatLng(pos.getLatitudeE6() / 1e6, pos.getLongitudeE6() / 1e6))
							.title(Integer.toString(idx))
							.snippet(String.format("%g, %g", pos.getLatitudeE6() / 1e6, pos.getLongitudeE6() / 1e6))
							.icon(BitmapDescriptorFactory.fromResource(R.drawable.marker_default)));
					mWaypointMarkers.add(idx, waypointMarker);
				} else {
					mWaypointMarkers.get(idx).setPosition((new LatLng(pos.getLatitudeE6() / 1e6, pos.getLongitudeE6() / 1e6)));
				}
			}
		}
	}

	private void setTabletTracking(boolean trackingEnabled) {
		// Disable the tracking function when tablet location unknown
		ToggleButton toggle = (ToggleButton) getActivity().findViewById(R.id.cameraPoiButton);
		if (toggle != null)
			toggle.setEnabled(trackingEnabled);
		if (toggle != null && !trackingEnabled)
			toggle.setChecked(false);

		// Disable the tracking function when tablet location unknown		
		toggle = (ToggleButton) getActivity().findViewById(R.id.rttButton); 
		if (toggle != null)
			toggle.setEnabled(trackingEnabled);
		if (toggle != null && !trackingEnabled)
			toggle.setChecked(false);

		toggle = (ToggleButton) getActivity().findViewById(R.id.followTabletButton); 
		if (toggle != null)
			toggle.setEnabled(trackingEnabled);
		if (toggle != null && !trackingEnabled)
			toggle.setChecked(false);
	}
	private GoogleMap.OnMarkerDragListener dragListener = new GoogleMap.OnMarkerDragListener() {

		@Override
		public void onMarkerDrag(Marker arg0) {
		}

		@Override
		public void onMarkerDragEnd(Marker arg0) {
			if (DEBUG) Log.d(TAG, "Drag ended: " + arg0.getId());
			// If path desired is what was dragged
			
			if (arg0.equals(mPathDesiredMarker)) {
				if (DEBUG) Log.d(TAG, "Path Desired");
				LatLng newPos = mPathDesiredMarker.getPosition();
				
				// TODO: check only in position hold tablet mode

				UAVObject pos = objMngr.getObject("PathDesired");
				if (pos == null)
					return;

				UAVObject home = objMngr.getObject("HomeLocation");
				if (home == null)
					return;

				double home_lat, home_lon, home_alt;
				home_lat = home.getField("Latitude").getDouble() / 10.0e6;
				home_lon = home.getField("Longitude").getDouble() / 10.0e6;
				home_alt = home.getField("Altitude").getDouble();

				// Get the home coordinates
				double T0, T1;
				T0 = home_alt+6.378137E6;
				T1 = Math.cos(home_lat * Math.PI / 180.0)*(home_alt+6.378137E6);

				// Get the NED coordinates
				double NED0, NED1;
				NED0 = (newPos.latitude - home_lat) * Math.PI / 180.0 * T0;
				NED1 = (newPos.longitude - home_lon) * Math.PI / 180.0 * T1;
				
				pos.getField("End").setDouble(NED0, 0);
				pos.getField("End").setDouble(NED1, 1);
				pos.updated();
			}
		}

		@Override
		public void onMarkerDragStart(Marker arg0) {
			// TODO Auto-generated method stub
			
		}
		
	};
	
	@Override
	public void onCreateContextMenu(ContextMenu menu, View v,
	                                ContextMenuInfo menuInfo) {
	    super.onCreateContextMenu(menu, v, menuInfo);
	    MenuInflater inflater = getActivity().getMenuInflater();
	    inflater.inflate(R.menu.map_click_actions, menu);
	}

	@Override
	public boolean onContextItemSelected(MenuItem item) {
	    switch (item.getItemId()) {
	        case R.id.map_action_jump_to_uav:
	            if (uavLocation != null) {
	            	mMap.animateCamera(CameraUpdateFactory.newLatLng(
	            			new LatLng(uavLocation.getLatitudeE6() / 1e6, uavLocation.getLongitudeE6() / 1e6)));
	            }
	            return true;
	        default:
	            return super.onContextItemSelected(item);
	    }
	}
	
	@Override
	public void onResume() {
		super.onResume();
		m.onResume();
	}

	@Override
	public void onPause() {
		super.onPause();
		m.onPause();
	}

	@Override
	public void onDestroy() {
		super.onDestroy();
		m.onDestroy();
	}

	@Override
	public void onLowMemory() {
		super.onLowMemory();
		m.onLowMemory();
	}
}
