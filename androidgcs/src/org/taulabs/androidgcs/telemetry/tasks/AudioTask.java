/**
 ******************************************************************************
 * @file       AudioTask.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      An @ref ITelemTask which generates audio alerts
 * @see        The GNU Public License (GPL) Version 3
 *
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
package org.taulabs.androidgcs.telemetry.tasks;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Observable;
import java.util.Observer;

import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectManager;

import android.content.Context;
import android.speech.tts.TextToSpeech;
import android.util.Log;

public class AudioTask implements ITelemTask, TextToSpeech.OnInitListener {

	final String TAG = AudioTask.class.getSimpleName();
	final boolean VERBOSE = false;
	final boolean DEBUG = true;

	private UAVObjectManager objMngr;
	private final List<UAVObject> listeningList = new ArrayList<UAVObject>();

	private TextToSpeech tts = null;
	private boolean ttsInit = false;

	@Override
	public void connect(UAVObjectManager o, Context context) {
		objMngr = o;

		ttsInit = false;
		tts = new TextToSpeech(context, this);

		// No objects registered at this point so watch for them
		objMngr.addNewObjectObserver(newObjObserver);
	}

	@Override
	public void disconnect() {
		unregisterAllObjects();
	}

	//! Register an object to inform this task on updates for logging
	private void registerObject(UAVObject obj) {
		synchronized(listeningList) {
			if (!listeningList.contains(obj)) {
				obj.addUpdatedObserver(objUpdatedObserver);
				listeningList.add(obj);
			}
		}
	}

	//! Unregister all objects from logging
	private void unregisterAllObjects() {
		synchronized(listeningList) {
			for (int i = 0; i < listeningList.size(); i++) {
				listeningList.get(i).removeUpdatedObserver(objUpdatedObserver);
			}
			listeningList.clear();
		}
	}

	//! Observer to catch when new objects or instances are registered
	private final Observer newObjObserver = new Observer() {
		@Override
		public void update(Observable observable, Object data) {
			UAVObject obj = (UAVObject) data;

			if (obj.getName().compareTo("FlightStatus") == 0) {
				if (DEBUG) Log.d(TAG, "Registered FlightStatus");
				registerObject(obj);
			}

			if (obj.getName().compareTo("SystemAlarms") == 0) {
				registerObject(obj);
			}

		}
	};

	//! Observer to catch when objects are updated
	private final Observer objUpdatedObserver = new Observer() {
		@Override
		public void update(Observable observable, Object data) {

			UAVObject obj = (UAVObject) data;

			if (DEBUG) Log.d(TAG, "Update received:" + obj.getName());

			if (ttsInit == false)
				return;

			if (obj.getName().compareTo("FlightStatus") == 0) {
				tts.speak("Flight Status" + obj.toString(), TextToSpeech.QUEUE_ADD, null);
			}

			if (obj.getName().compareTo("SystemAlarms") == 0) {
				tts.speak("System Alarms " + obj.toString(), TextToSpeech.QUEUE_ADD, null);
			}
}
	};

	@Override
	public void onInit(int status) {
		ttsInit = true;
		tts.setLanguage(Locale.US);
		tts.speak("TTS running", TextToSpeech.QUEUE_ADD, null);
	}


}
