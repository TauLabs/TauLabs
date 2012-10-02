package org.openpilot.androidgcs.fragments;

import org.junit.Assert;
import org.openpilot.androidgcs.R;
import org.openpilot.uavtalk.UAVObject;
import org.openpilot.uavtalk.UAVObjectManager;
import org.osmdroid.ResourceProxy;
import org.osmdroid.views.MapView;
import org.osmdroid.views.overlay.MyLocationOverlay;

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
	protected ResourceProxy mResourceProxy;

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

		Assert.assertNotNull(mOsmv);

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

}
