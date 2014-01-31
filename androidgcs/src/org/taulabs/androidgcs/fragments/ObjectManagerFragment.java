/**
 ******************************************************************************
 * @file       ObjectManagerFragment.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Base class for all fragments that use the UAVObjectManager.  This
 *             supports all the extensions the ObjectManagerActivity does, namely
 *             access to the UAVObjectManager and callbacks in the UI thread for
 *             object updates.
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

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Observable;
import java.util.Observer;
import java.util.Set;

import org.taulabs.androidgcs.ObjectManagerActivity;
import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectManager;

import android.app.Activity;
import android.app.Fragment;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;

public abstract class ObjectManagerFragment extends Fragment {

	private static final String TAG = ObjectManagerFragment.class.getSimpleName();
	private static final int LOGLEVEL = 0;
	private static boolean VERBOSE = LOGLEVEL > 2;
//	private static boolean WARN = LOGLEVEL > 1;
	private static final boolean DEBUG = LOGLEVEL > 0;

	UAVObjectManager objMngr;
	
	abstract protected String getDebugTag();

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		if (DEBUG) Log.d(TAG, "onCreate: " + getDebugTag());
		// For an activity this registers against the telemetry service intents.  Fragments must be notified by their
		// parent activity
	}

	/**
	 * Attach to the parent activity so it can notify us when the connection
	 * changed
	 */
    @Override
    public void onAttach(Activity activity) {
    	super.onAttach(activity);
    	if (DEBUG) Log.d(TAG,"onAttach: " + getDebugTag());

        ObjectManagerActivity castActivity = null;
        try {
        	castActivity = (ObjectManagerActivity)getActivity();
        } catch (ClassCastException e) {
        	throw new android.app.Fragment.InstantiationException(
        			"Attaching a ObjectManagerFragment to an activity failed because the parent activity is not a ObjectManagerActivity",
        			e);
        }
        castActivity.addOnConnectionListenerFragment(this);
    }
    
    @Override
    public void onDetach() {
    	super.onDetach();
    	if (DEBUG) Log.d(TAG,"onDetach: " + getDebugTag());
    }
    
    @Override
	public void onStop() {
    	super.onStop();
    	if (DEBUG) Log.d(TAG, "onStop: " + getDebugTag());
    }
    
    private boolean resumed = false;

    @Override
    public void onPause() {
    	super.onPause();
    	if (DEBUG) Log.d(TAG, "onPause: " + getDebugTag());
    	resumed = false;
    }
    
    public void onResume() {
    	super.onResume();
    	if (DEBUG) Log.d(TAG, "onResume: " + getDebugTag());
    	resumed = true;
    }

    /**
     * When the fragment is destroyed we must remove all the callbacks
     * as the handlers will no longer be valid.
     */
    synchronized public void onDestroy() {
    	super.onDestroy();
    	
    	if (DEBUG) Log.d(TAG, "onDestroy: " + getDebugTag());
    	
    	synchronized(listeners) {
    		Set<Observer> s = listeners.keySet();
    		Iterator<Observer> i = s.iterator();
    		while (i.hasNext()) {
    			Observer o = i.next();
    			UAVObject obj = listeners.get(o);
    			obj.removeUpdatedObserver(o);

    			ObjectyUpdatedObserver oo = (ObjectyUpdatedObserver) o;

    			if (VERBOSE) Log.v(getDebugTag(), "Removed listener for " + obj.getName()+ " observer: " + oo.my_count);
    		}

    		if (VERBOSE) Log.v(TAG, "onDestroy deleted " + listeners.size() + " listeneners");
    		listeners.clear();
    	}
		
    	// Unfortunate there still seems to be the ability to have race conditions
    	// so we must also make sure no updates are pass through after this point
    	disableUpdates = true;

    	if (objMngr != null)
    		onDisconnected();
		
    }

	// The below methods should all be called by the parent activity at the appropriate times
	synchronized public void onConnected(UAVObjectManager objMngr) {
		this.objMngr = objMngr;
		if (DEBUG) Log.d(TAG,"onConnected: " + getDebugTag());
	}

	synchronized public void onDisconnected() {
		objMngr = null;
		if (DEBUG) Log.d(TAG,"onDisconnected: " + getDebugTag());
	}

	/**
	 * Called whenever any objects subscribed to via registerObjects
	 * is updated.
	 */
	protected void objectUpdated(UAVObject obj) {
	}
	
	/**
	 * Called whenever any objects subscribed to via registerObjects
	 * is updated and the UI is valid to update.
	 */
	protected void objectUpdatedUI(UAVObject obj) {
	}
	
	//! Handler that posts messages from object updates
	final Handler uavobjHandler = new Handler();

	// Flag that indicates observers should have been disconnected so any
	// pending updates should be quashed
	public boolean disableUpdates = false;

	// Cheap helper for tracking the observers
	static int observer_count = 0;

	//! Observer to notify the fragment of an update
	private class ObjectyUpdatedObserver implements Observer  {
		UAVObject obj;
		public int my_count;
		
		ObjectyUpdatedObserver(UAVObject obj) { this.obj = obj; my_count = observer_count; observer_count++; };
		@Override
		public void update(Observable observable, Object data) {
			uavobjHandler.post(new Runnable() {
				@Override
				public void run() {
					if (disableUpdates) {
						Log.w(getDebugTag(), "Got an update for " + obj.getName() + " after it should have been disabled");
						return;
					}
					// These assertions catch bugs in our lifecycle process
					if (objMngr == null) {
						if (DEBUG) Log.d(getDebugTag(), "Null object manager update for " + obj.getName() + " observer: " + my_count);
					}
					
					// Send the notifications
					objectUpdated(obj);
					if (resumed)
						objectUpdatedUI(obj);
				}
			});
		}
	};
	
	//! Maintain a list of all the UAVObject listeners for this fragment
	private HashMap<Observer, UAVObject> listeners = new HashMap<Observer, UAVObject>();

	/**
	 * Register an activity to receive updates from this object
	 * @param object The object the activity should listen to updates from
	 * the objectUpdated() method will be called in the original UI thread
	 */
	protected void registerObjectUpdates(UAVObject object) {
		ObjectyUpdatedObserver o = new ObjectyUpdatedObserver(object);
		if (VERBOSE) Log.v(getDebugTag(), "registerObjectUpdates for " + object + " observer: " + o.my_count);
		listeners.put(o,  object);
		object.addUpdatedObserver(o);
	}

	/**
	 * Helper method to register array of objects
	 */
	protected void registerObjectUpdates(List<List<UAVObject>> objects) {
		for (int i = 0; i < objects.size(); i++) {
			List<UAVObject> inner = objects.get(i);
			for (int j = 0; j < inner.size(); j++)
				registerObjectUpdates(inner.get(j));
		}
	}

}
