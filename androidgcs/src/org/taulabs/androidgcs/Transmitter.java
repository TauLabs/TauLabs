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

import org.taulabs.androidgcs.R;
import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectField;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.ToggleButton;

public class Transmitter extends ObjectManagerActivity {

	private final static String TAG = Transmitter.class.getSimpleName();
	private final TwoWayHashmap <String, Integer>modesToId = new TwoWayHashmap<String, Integer>();

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.transmitter);

		((RadioGroup) findViewById(R.id.modeSelector1)).setOnCheckedChangeListener(ToggleListener);

		modesToId.add("PositionHold",R.id.positionHoldButton);
		modesToId.add("ReturnToHome",R.id.rthButton);
		modesToId.add("ReturnToTablet",R.id.rttButton);
		modesToId.add("PathPlanner",R.id.pathPlannerButton);
		modesToId.add("FollowMe",R.id.followTabletButton);
		modesToId.add("Land",R.id.landButton);
	}

	@Override
	void onOPConnected() {
		super.onOPConnected();

		// Get the current tablet mode desired to make sure screen reflects what the
		// UAV is doing when we jump out of this activity
		UAVObject obj = objMngr.getObject("TabletInfo");
		UAVObjectField field;

    	if (obj != null && (field = obj.getField("TabletModeDesired")) != null) {
    		String mode = field.getValue().toString();
    		Log.d(TAG, "Connected and mode is: " + mode);

    		Integer id = modesToId.getForward(mode);
    		if (id == null)
    			Log.e(TAG, "Unknown mode");
    		else
    			onToggle(findViewById(id));
    	}

    	obj = objMngr.getObject("FlightStatus");
    	if (obj != null)
        	registerObjectUpdates(obj);
    	obj.updateRequested();

	}

	@Override
	protected void objectUpdated(UAVObject obj) {
		if (obj.getName().compareTo("FlightStatus") == 0) {
			UAVObjectField field = obj.getField("FlightMode");
			if (field != null) {
				TextView text = (TextView) findViewById(R.id.flightMode);
				text.setText(field.getValue().toString());
			}

			field = obj.getField("Armed");
			if (field != null) {
				TextView text = (TextView) findViewById(R.id.armedStatus);
				text.setText(field.getValue().toString());
			}
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

}
