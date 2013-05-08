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
import org.taulabs.uavtalk.UAVObjectField;
import org.taulabs.uavtalk.UAVObjectManager;

import android.content.Context;
import android.speech.tts.TextToSpeech;

public class AudioTask implements ITelemTask, TextToSpeech.OnInitListener, TextToSpeech.OnUtteranceCompletedListener {

	/* Debugging information */
	final String TAG = AudioTask.class.getSimpleName();
	final boolean VERBOSE = false;
	final boolean DEBUG = false;

	/* IDs for some of the messages */
	final static String GPS_ALARM_MSG_ID     = "GPS_ALARM_MSG_ID";
	final static String GPS_ALARM_CLR_MSG_ID = "GPS_ALARM_CLR_MSG_ID";
	final static String UNTAGGED_MSG_ID      = "UNTAGGED_MSG_ID";

	/* Private variables */
	private UAVObjectManager objMngr;
	private final List<UAVObject> listeningList = new ArrayList<UAVObject>();

	private TextToSpeech tts = null;
	private boolean ttsInit = false;

	/* Track some pending announcements to stop them backing up */
	private boolean gps_alarm_pending = false;
	private boolean gps_clear_pending = false;

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
		ttsInit = false;
		if (tts != null) {
			tts.shutdown();
			tts = null;
		}
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
				registerObject(obj);
			}

			if (obj.getName().compareTo("SystemAlarms") == 0) {
				registerObject(obj);
			}

			if (obj.getName().compareTo("WaypointActive") == 0) {
				registerObject(obj);
			}

		}
	};

	//! Observer to catch when objects are updated
	private final Observer objUpdatedObserver = new Observer() {
		@Override
		public void update(Observable observable, Object data) {

			UAVObject obj = (UAVObject) data;

			if (ttsInit == false || tts == null)
				return;

			if (obj.getName().compareTo("FlightStatus") == 0) {
				flightStatusUpdated(obj);
			}

			if (obj.getName().compareTo("SystemAlarms") == 0) {
				alarmsUpdated(obj);
			}

			if (obj.getName().compareTo("WaypointActive") == 0) {
				waypointActiveUpdated(obj);
			}
		}
	};

	@Override
	public void onInit(int status) {
		if (tts == null && status != TextToSpeech.SUCCESS)
			return;
		ttsInit = true;
		tts.setLanguage(Locale.US);
	}

	List<String> alarmValues = null;

	private void alarmsUpdated(UAVObject obj) {
		UAVObjectField alarm = obj.getField("Alarm");
		List<String> alarmNames = alarm.getElementNames();

		if (alarmValues == null) {
			alarmValues = new ArrayList<String>();
			for (int i = 0; i < alarm.getNumElements(); i++) {
				// Initialize all alarms to OK so any warnings are annouced
				alarmValues.add("OK");
			}
		}

		int severity = 0;
		for (int i = 0; i < alarm.getNumElements(); i++) {
			if (alarmNames.get(i).compareTo("EventSystem")==0)
				continue;

			int thisSeverity = (int) alarm.getDouble(i);
			String newAlarm = alarm.getValue(i).toString();
			if (thisSeverity > severity)
				severity = thisSeverity;

			// When GPS is near threshold this can start toggling
			// very quickly.  Filter to prevent backing up the queue.
			if (alarmNames.get(i).compareTo("GPS")==0) {
				if (thisSeverity == 1 && gps_clear_pending)
					continue;
				if (thisSeverity > 1 && gps_alarm_pending)
					continue;
			}

			// If alarm is set and previously wasn't described
			if (thisSeverity > 1 && alarmValues.get(i).compareTo(newAlarm) != 0)
				tts.speak(alarmNames.get(i) + " " + newAlarm, TextToSpeech.QUEUE_ADD, null);
			else if (thisSeverity == 1 && alarmValues.get(i).compareTo(newAlarm) != 0)
				tts.speak(alarmNames.get(i) + " alarm cleared", TextToSpeech.QUEUE_ADD, null);
			alarmValues.set(i,newAlarm);
		}
	}

	private String armed = "Disarmed";
	private String flightMode = "Stabilized1";
	private void flightStatusUpdated(UAVObject obj) {

		// Announce when armed or disarmed
		String newArmed = obj.getField("Armed").getValue().toString();
		if (newArmed.compareTo(armed) != 0) {
			armed = newArmed;
			if (armed.compareTo("Arming") != 0)
				tts.speak(newArmed, TextToSpeech.QUEUE_ADD, null);
		}

		// Announce change in flight mode
		String newFlightMode = obj.getField("FlightMode").getValue().toString();
		if (newFlightMode.compareTo(flightMode) != 0) {
			flightMode = newFlightMode;
			tts.speak("Flight Mode " + flightMode, TextToSpeech.QUEUE_ADD, null);
		}
	}

	// Announce the activated waypoint when in PathPlanner mode
	private void waypointActiveUpdated(UAVObject obj) {
		int index = (int) obj.getField("Index").getDouble();
		if (flightMode.compareTo("PathPlanner") == 0) {
			tts.speak("Activated waypoint " + index, TextToSpeech.QUEUE_ADD, null);
		}
	}

	@Override
	public void onUtteranceCompleted(String utteranceId) {
		if (utteranceId.compareTo(GPS_ALARM_MSG_ID) == 0)
			gps_alarm_pending = false;
		if (utteranceId.compareTo(GPS_ALARM_CLR_MSG_ID) == 0)
			gps_clear_pending = false;
	}
}
