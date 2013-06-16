package org.taulabs.androidgcs.fragments;

import org.taulabs.androidgcs.R;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.google.android.gms.maps.MapView;

public class Map extends ObjectManagerFragment {
	MapView m;

	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
			Bundle savedInstanceState) {
		// inflate and return the layout
		View v = inflater.inflate(R.layout.map_fragment, container, false);
		m = (MapView) v.findViewById(R.id.mapView);
		m.onCreate(savedInstanceState);

		return v;
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
