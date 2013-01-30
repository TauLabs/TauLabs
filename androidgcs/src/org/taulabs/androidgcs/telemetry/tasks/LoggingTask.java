/**
 ******************************************************************************
 * @file       LoggingTask.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      An @ref ITelemTask which generates logs
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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Observable;
import java.util.Observer;

import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectField;
import org.taulabs.uavtalk.UAVObjectManager;
import org.taulabs.uavtalk.UAVTalk;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

public class LoggingTask implements ITelemTask {

	final String TAG = LoggingTask.class.getSimpleName();
	final boolean VERBOSE = false;
	final boolean DEBUG = false;

	private UAVObjectManager objMngr;
	private final List<UAVObject> listeningList = new ArrayList<UAVObject>();
	private boolean loggingActive = false;
	private boolean headerWritten = false;


	private File file;
	private FileOutputStream fileStream;
	private UAVTalk uavTalk;

	private int writtenBytes;
	private int writtenObjects;

	@Override
	public void connect(UAVObjectManager o, Context context) {
		objMngr = o;

		// When new objects are registered, ensure we listen
		// to them
		o.addNewObjectObserver(newObjObserver);
		o.addNewInstanceObserver(newObjObserver);

		// Register all existing objects
		List<List<UAVObject>> objects = objMngr.getObjects();
		for(int i = 0; i < objects.size(); i++)
			for(int j = 0; j < objects.get(i).size(); j++)
				registerObject(objects.get(i).get(j));

		// For now default to starting to log
		startLogging();
	}

	@Override
	public void disconnect() {
		endLogging();
		objMngr.deleteNewObjectObserver(newObjObserver);
		objMngr.deleteNewInstanceObserver(newObjObserver);
		unregisterAllObjects();
	}

	public int getWrittenBytes() {
		return writtenBytes;
	}
	public int getWrittenObjects() {
		return writtenObjects;
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

	//! Write an updated object to the log file
	private void logObject(UAVObject obj) {
		if (loggingActive && headerWritten) {
			if (VERBOSE) Log.v(TAG,"Updated: " + obj.toString());
			try {
				long time = System.currentTimeMillis();
				fileStream.write((byte)(time & 0xff));
				fileStream.write((byte)((time & 0x0000ff00) >> 8));
				fileStream.write((byte)((time & 0x00ff0000) >> 16));
				fileStream.write((byte)((time & 0xff000000) >> 24));

				long size = obj.getNumBytes();
				fileStream.write((byte)(size & 0x00000000000000ffl) >> 0);
				fileStream.write((byte)(size & 0x000000000000ff00l) >> 8);
				fileStream.write((byte)(size & 0x0000000000ff0000l) >> 16);
				fileStream.write((byte)(size & 0x00000000ff000000l) >> 24);
				fileStream.write((byte)(size & 0x000000ff00000000l) >> 32);
				fileStream.write((byte)(size & 0x0000ff0000000000l) >> 40);
				fileStream.write((byte)(size & 0x00ff000000000000l) >> 48);
				fileStream.write((byte)(size & 0xff00000000000000l) >> 56);

				uavTalk.sendObject(obj, false, false);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			writtenBytes += obj.getNumBytes();
			writtenObjects ++;
		}
	}

	//! Open a file and start logging
	private boolean startLogging() {
		File root = Environment.getExternalStorageDirectory();

		// Make the directory if it doesn't exist
		File logDirectory = new File(root, "/TauLabs");
		logDirectory.mkdirs();

		Date d = new Date();
		String date = (new SimpleDateFormat("yyyyMMdd_hhmmss")).format(d);
		String fileName = "log_" + date + ".opl";

		file = new File(logDirectory, fileName);
		if (DEBUG) Log.d(TAG, "Trying for file: " + file.getAbsolutePath());
		try {
			if (root.canWrite()){
				fileStream = new FileOutputStream(file);
				uavTalk = new UAVTalk(null, fileStream, objMngr);
				writtenBytes = 0;
				writtenObjects = 0;
			} else {
				Log.e(TAG, "Unwriteable address");
				loggingActive = false;
				return loggingActive;
			}
		} catch (IOException e) {
			Log.e(TAG, "Could not write file " + e.getMessage());
			loggingActive = false;
			return loggingActive;

		}

		loggingActive = file.canWrite();

		// Add listener to write the version header when available
		UAVObject firmwareIapObj = objMngr.getObject("FirmwareIAPObj");
		if (firmwareIapObj != null) {
			firmwareIapObj.addUpdatedObserver(firmwareIapUpdated);
		}

		return loggingActive;
	}

	//! Close the file and end logging
	private boolean endLogging() {
		loggingActive = false;

		if (DEBUG) Log.d(TAG, "Stop logging");
		try {
			if (fileStream != null) {
				fileStream.close();
				fileStream = null;
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return true;
	}

	//! Observer to catch when new objects or instances are registered
	private final Observer newObjObserver = new Observer() {
		@Override
		public void update(Observable observable, Object data) {
			UAVObject obj = (UAVObject) data;
			registerObject(obj);
		}
	};

	//! Observer to catch when objects are updated
	private final Observer objUpdatedObserver = new Observer() {
		@Override
		public void update(Observable observable, Object data) {
			UAVObject obj = (UAVObject) data;
			logObject(obj);
		}
	};

	public class Stats {
		public boolean loggingActive;
		public int writtenBytes;
		public int writtenObjects;
	};

	//! Return an object with the logging stats
	public Stats getStats() {
		Stats s = new Stats();
		s.loggingActive = loggingActive;
		s.writtenBytes = writtenBytes;
		s.writtenObjects = writtenObjects;
		return s;
	}

	private final Observer firmwareIapUpdated = new Observer() {

		@Override
		public void update(Observable observable, Object data) {
			UAVObject firmwareIapObj = (UAVObject) data;
			if (firmwareIapObj == null)
				return;

			UAVObjectField description = firmwareIapObj.getField("Description");
			if (description == null || description.getNumElements() < 100) {
				Log.d(TAG, "Failed to determine description for logging");
			} else {
				final int GITASH_SIZE_USED = 4;
				final int UAVOHASH_SIZE_USED = 8;

				String gitHash = new String();
				String uavoHash = new String();

				for (int i = 0; i < GITASH_SIZE_USED; i++) {
					gitHash += String.format("%02x", (int) description.getDouble(7 - i));
				}
				for (int i = 0; i < UAVOHASH_SIZE_USED; i++) {
					uavoHash += String.format("%02x", (int) description.getDouble(i + 60));
				}

				if (VERBOSE) Log.v(TAG, "Git hash: " + gitHash);
				if (VERBOSE) Log.v(TAG, "UAVO hash: " + uavoHash);

				try {
					fileStream.write(String.format("Tau Labs git hash: %s\n%s\n##\n", gitHash, uavoHash).getBytes());
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

			headerWritten = true;
			firmwareIapObj.removeUpdatedObserver(this);
		}

	};

}
