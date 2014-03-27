/**
 ******************************************************************************
 * @file       Transmitter.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Activity for controlling the UAV from a phone or tablet
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

import java.util.HashMap;
import java.util.Map;

import org.taulabs.androidgcs.drawer.NavDrawerActivityConfiguration;
import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectField;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.RadioGroup;
import android.widget.ToggleButton;

public class TabletControl extends ObjectManagerActivity {

	private final static String TAG = TabletControl.class.getSimpleName();
	private final TwoWayHashmap <String, Integer>modesToId = new TwoWayHashmap<String, Integer>();

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		((RadioGroup) findViewById(R.id.modeSelector1)).setOnCheckedChangeListener(ToggleListener);

		modesToId.add("PositionHold",R.id.positionHoldButton);
		modesToId.add("ReturnToHome",R.id.rthButton);
		modesToId.add("ReturnToTablet",R.id.rttButton);
		modesToId.add("PathPlanner",R.id.pathPlannerButton);
		modesToId.add("FollowMe",R.id.followTabletButton);
		modesToId.add("Land",R.id.landButton);
	}
	
	@Override
	protected NavDrawerActivityConfiguration getNavDrawerConfiguration() {
		NavDrawerActivityConfiguration navDrawer = getDefaultNavDrawerConfiguration();
		navDrawer.setMainLayout(R.layout.tablet_control);
		return navDrawer;
	}
	
	@Override
	void onConnected() {
		super.onConnected();

		// Get the current tablet mode desired to make sure screen reflects what the
		// UAV is doing when we jump out of this activity
		UAVObject obj = objMngr.getObject("TabletInfo");
		UAVObjectField field;

		// Update the active mode
    	if (obj != null && (field = obj.getField("TabletModeDesired")) != null) {
    		String mode = field.getValue().toString();
    		Log.d(TAG, "Connected and mode is: " + mode);

    		Integer id = modesToId.getForward(mode);
    		if (id == null)
    			Log.e(TAG, "Unknown mode");
    		else
    			onToggle(findViewById(id));
    	}

    	// Update the POI track button
    	if (obj != null && (field = obj.getField("POI")) != null) {
    		boolean poiTrack = field.getValue().toString().compareTo("True") == 0;
    		ToggleButton poiTrackButton = (ToggleButton) findViewById(R.id.cameraPoiButton);
    		if (poiTrackButton != null)
    			poiTrackButton.setChecked(poiTrack);
    	}

	}


	//! Process the changes in the mode selector and pass that information to TabletInfo
	final RadioGroup.OnCheckedChangeListener ToggleListener = new RadioGroup.OnCheckedChangeListener() {
        @Override
        public void onCheckedChanged(final RadioGroup radioGroup, final int i) {
        	Log.d("Transmitter", "Toggled");
            for (int j = 0; j < radioGroup.getChildCount(); j++) {
                final ToggleButton view = (ToggleButton) radioGroup.getChildAt(j);
                view.setChecked(view.getId() == i);
            }

            if (objMngr != null) {
            	UAVObject obj = objMngr.getObject("TabletInfo");
            	if (obj == null)
            		return;
            	UAVObjectField field = obj.getField("TabletModeDesired");
            	if (field == null)
            		return;

            	String mode = modesToId.getBackward(i);
            	if (mode != null) {
            		Log.i(TAG, "Selecting mode: " + mode);
            		field.setValue(mode);
            	} else
            		Log.e(TAG, "Unknown mode for this button");

            	obj.updated();
            }
        }
    };

    public void onToggle(View view) {
    	ToggleButton v = (ToggleButton) view;
    	v.setChecked(true);
        ((RadioGroup)view.getParent()).check(view.getId());
    }

    private class TwoWayHashmap<K extends Object, V extends Object> {
    	private final Map<K,V> forward = new HashMap<K, V>();
    	private final Map<V,K> backward = new HashMap<V, K>();
    	public synchronized void add(K key, V value) {
    		forward.put(key, value);
    		backward.put(value, key);
    	}
    	public synchronized V getForward(K key) {
    		return forward.get(key);
    	}
    	public synchronized K getBackward(V key) {
    		return backward.get(key);
    	}
    }
    
    public void onPoiToggle(View view)
    {
    	ToggleButton toggle = (ToggleButton) view;
    	if (toggle == null)
    		return;

		// Set the tablet POI tracking mode based on check box
		if (objMngr != null) {
			UAVObject obj = objMngr.getObject("TabletInfo");
			if (obj == null)
				return;
			UAVObjectField field = obj.getField("POI");
			if (field == null)
				return;
			if (toggle.isChecked())
				field.setValue("True");
			else
				field.setValue("False");
			obj.updated();
		}
    };

}
