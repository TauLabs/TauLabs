/**
 ******************************************************************************
 * @file       SmartSave.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Provides a handler to provide robust apply and save for settings
 *             and updating of the UI when the object changes.
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

package org.taulabs.androidgcs.util;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Observable;
import java.util.Observer;
import java.util.Set;

import junit.framework.Assert;

import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectManager;

import android.app.Activity;
import android.graphics.LightingColorFilter;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;

public class SmartSave {

	private final static String TAG = SmartSave.class.getSimpleName();
	private final static boolean DEBUG = false;
	private final Activity parentActivity;

	//! Create a smart save button attached to the object manager and an apply and ave button
	public SmartSave(UAVObjectManager objMngr, Activity parent, UAVObject obj, Button saveButton, Button applyButton, Button loadButton) {
		Assert.assertNotNull(objMngr);
		this.objMngr = objMngr;
		this.parentActivity = parent;
		this.applyBtn = applyButton;
		this.obj = obj;

		Assert.assertNotNull(objMngr);
		Assert.assertNotNull(obj);

		obj.addUpdatedObserver(ObjectUpdated);

		controlFieldMapping = new HashMap<ObjectFieldMappable,FieldPairing>();

		if (saveButton != null) {
			saveBtn = saveButton;
			saveBtn.setOnClickListener(new OnClickListener() {
				@Override
				public void onClick(View v) {
					if (saveSettings())
						saveBtn.getBackground().setColorFilter(new LightingColorFilter(0x11111111, 0xFF00FF00));
					else
						saveBtn.getBackground().setColorFilter(new LightingColorFilter(0x11111111, 0xFFFF0000));

				}
			});
		} else
			saveBtn = null;

		if (applyButton != null) {
			applyBtn = applyButton;
			applyBtn.setOnClickListener(new OnClickListener() {
				@Override
				public void onClick(View v) {
					if (applySettings())
						applyBtn.getBackground().setColorFilter(new LightingColorFilter(0x11111111, 0xFF00FF00));
					else
						applyBtn.getBackground().setColorFilter(new LightingColorFilter(0x11111111, 0xFFFF0000));
				}
			});
		} else
			applyBtn = null;

		if (loadButton != null) {
			loadBtn = loadButton;
			loadBtn.setOnClickListener(new OnClickListener() {
				@Override
				public void onClick(View v) {
					if (loadSettings())
						loadBtn.getBackground().setColorFilter(new LightingColorFilter(0x11111111, 0xFF00FF00));
					else
						loadBtn.getBackground().setColorFilter(new LightingColorFilter(0x11111111, 0xFFFF0000));
				}
			});
		} else
			loadBtn = null;
	}

	//! Disconnect any listeners when this object is destroyed
	public void disconnect() {
		obj.removeUpdatedObserver(ObjectUpdated);
	}

	/**
	 * Add a control to this SmartSave object which maps between a particular control
	 * and a UAVO field.
	 * @param control The control to associate with
	 * @param fieldName The name of the UAVO field
	 * @param fieldIndex The index of the UAVO field
	 */
	public void addControlMapping(ObjectFieldMappable control, String fieldName, int fieldIndex) {
		FieldPairing pairing = new FieldPairing(fieldName, fieldIndex);
		controlFieldMapping.put(control, pairing);
		control.setOnChangedListener(markDirty);
	}

	//! Update the settings in the UI from the mappings
	public void refreshSettingsDisplay() {
		if (DEBUG) Log.d(TAG, "Refreshing display");

		Set<ObjectFieldMappable> keys = controlFieldMapping.keySet();
		Iterator<ObjectFieldMappable> iter = keys.iterator();
		while(iter.hasNext()) {
			ObjectFieldMappable mappable = iter.next();
			FieldPairing field = controlFieldMapping.get(mappable);
			mappable.setValue(field.getValue(obj));
		}
	}

	/**
	 * Call applySettings() and then save them
	 * @return True if the save succeeded or false otherwise
	 */
	private boolean saveSettings() {
		/*
		 * 1. Apply settings
		 * 2. Install listener on object persistence
		 * 3. Send save operation
		 * 4. Wait for completed
		 * 5. Remove listener
		 */

		// 1. Apply settings
		if (!applySettings()) {
			if (DEBUG) Log.d(TAG, "Failed to apply settings");
			return false;
		}

		UAVObject persistence = objMngr.getObject("ObjectPersistence");
		Assert.assertNotNull(persistence);

		synchronized(ObjectPersistenceUpdated) {
			// 2. Install listener
			persistence.addUpdatedObserver(ObjectPersistenceUpdated);

			// 3. Send Load operation
			persistenceUpdated = false;
			Long objId = obj.getObjID();
			if (DEBUG) Log.d(TAG, "Saving object ID: " + objId);
			persistence.getField("ObjectID").setValue(objId);
			persistence.getField("Operation").setValue("Save");
			persistence.getField("Selection").setValue("SingleObject");
			persistence.updated();

			// 4. Wait for ack
			try {
				ObjectPersistenceUpdated.wait(1000);
			} catch (InterruptedException e) {
				if (DEBUG) Log.d(TAG ,"No ack for object persistence");
				persistence.removeUpdatedObserver(ObjectPersistenceUpdated);
				return false;
			}

			// 5. Remove listener
			persistence.removeUpdatedObserver(ObjectPersistenceUpdated);
		}

		return persistenceUpdated;
	}

	/**
	 * Robustly apply the settings to the UAV
	 * @return True if the apply is ack'd, False if not
	 */
	private boolean applySettings() {
		/*
		 * 1. Set the values from the fields into the object
		 * 2. Install the listener on the object
		 * 3. Update object
		 * 4. Wait for the acknowledgment or timeout
		 * 5. Uninstall the listener
		 */

		// 1. Set the fields in the object from the UI
		Set<ObjectFieldMappable> keys = controlFieldMapping.keySet();
		Iterator<ObjectFieldMappable> iter = keys.iterator();
		while(iter.hasNext()) {
			ObjectFieldMappable mappable = iter.next();
			FieldPairing field = controlFieldMapping.get(mappable);
			field.setValue(obj,mappable.getValue());
		}

		synchronized(ApplyCompleted) {
			if (DEBUG) Log.d(TAG, "Sending apply");

			// 2. Install the listener on the object
			obj.addTransactionCompleted(ApplyCompleted);

			// 3. Update the object
			objectUpdated = false;
			obj.updated();

			// 4. Wait for acknowledgment
			try {
				ApplyCompleted.wait(1000);
			} catch (InterruptedException e) {
				if (DEBUG) Log.d(TAG ,"Apply failed");
				obj.removeTransactionCompleted(ApplyCompleted);
				return false;
			}

			if (DEBUG) Log.d(TAG ,"Apply succeeded");

			// 5. Uninstall the listener
			obj.removeTransactionCompleted(ApplyCompleted);
		}

		return objectUpdated;
	}

	/**
	 * Robustly apply the settings to the UAV
	 * @return True if the apply is ack'd, False if not
	 */
	private boolean loadSettings() {
		/*
		 * 1. Update object
		 * 2. Install listener on object persistence
		 * 3. Send save operation
		 * 4. Wait for completed
		 * 5. Remove listener
		 * 6. Request update
		 * 7. Wait for completion
		 */

		UAVObject persistence = objMngr.getObject("ObjectPersistence");
		Assert.assertNotNull(persistence);

		synchronized(ObjectPersistenceUpdated) {
			// 2. Install listener
			persistence.addUpdatedObserver(ObjectPersistenceUpdated);

			// 3. Send save operation
			persistenceUpdated = false;
			Long objId = obj.getObjID();
			if (DEBUG) Log.d(TAG, "Load object ID: " + objId);
			persistence.getField("ObjectID").setValue(objId);
			persistence.getField("Operation").setValue("Load");
			persistence.getField("Selection").setValue("SingleObject");
			persistence.updated();

			// 4. Wait for the update
			try {
				ObjectPersistenceUpdated.wait(1000);
			} catch (InterruptedException e) {
				persistence.removeUpdatedObserver(ObjectPersistenceUpdated);
				return false;
			}

			// 5. Remove the listener
			persistence.removeUpdatedObserver(ObjectPersistenceUpdated);
		}

		// Only continue if the load worked
		if (persistenceUpdated == false)
			return false;

		synchronized(ApplyCompleted) {
			if (DEBUG) Log.d(TAG, "Requesting update");

			// Install the listener on the object
			obj.addTransactionCompleted(ApplyCompleted);

			// 6. Update the object
			objectUpdated = false;
			obj.updateRequested();

			// 7. Wait for acknowledgment
			try {
				ApplyCompleted.wait(1000);
			} catch (InterruptedException e) {
				if (DEBUG) Log.d(TAG ,"Apply failed");
				obj.removeTransactionCompleted(ApplyCompleted);
				return false;
			}

			if (DEBUG) Log.d(TAG ,"Apply succeeded");

			// Uninstall the listener
			obj.removeTransactionCompleted(ApplyCompleted);
		}

		return objectUpdated;
	}

	//! Private class to store the field mapping information
	private class FieldPairing {
		FieldPairing(String fieldName, int fieldIndex) {
			this.fieldName = fieldName;
			this.fieldIndex = fieldIndex;
		}

		//! Update the field in the UAVO
		void setValue(UAVObject obj, double value) {
			Assert.assertNotNull(obj);
			obj.getField(fieldName).setDouble(value,fieldIndex);
		}

		//! Get the value from the UAVO field
		double getValue(UAVObject obj) {
			double val = obj.getField(fieldName).getDouble(fieldIndex);
			if (DEBUG) Log.d(TAG, "Getting value from: " + fieldName + " " + val);
			return obj.getField(fieldName).getDouble(fieldIndex);
		}

		//! Cache the name of the field
		private final String fieldName;

		//! Cache the field index
		private final int fieldIndex;
	}

	//! Installed on monitored object to know when an object is updated
	private final Observer ApplyCompleted = new Observer() {
		@Override
		public void update(Observable observable, Object data) {
			synchronized(this) {
				objectUpdated = true;
				notify();
			}
		}
	};

	//! Installed on monitored object to know when an object is updated
	private final Observer ObjectUpdated = new Observer() {
		@Override
		public void update(Observable observable, Object data) {
			if (DEBUG) Log.d(TAG, "Object updated");
			parentActivity.runOnUiThread(new Runnable() {
				@Override
				public void run() {
					refreshSettingsDisplay();
				}
			});
		}
	};

	//! Installed on object persistence to know when save completes
	private final Observer ObjectPersistenceUpdated = new Observer() {
		@Override
		public void update(Observable observable, Object data) {
			if (DEBUG) Log.d(TAG, "Object persistence updated");
			synchronized(this) {
				persistenceUpdated = true;
				notify();
			}
		}
	};

	//! When the fields change this will reset the button to default appearance
	private final Runnable markDirty = new Runnable() {
		@Override
		public void run() {
			Log.d(TAG, "Marking dirty");
			if (applyBtn != null)
				applyBtn.getBackground().setColorFilter(null);
			if (loadBtn != null)
				loadBtn.getBackground().setColorFilter(null);
			if (saveBtn != null)
				saveBtn.getBackground().setColorFilter(null);
		}
	};

	//! Map of all the UAVO field <-> control mappings.  The key is the control.
	private final Map<ObjectFieldMappable,FieldPairing> controlFieldMapping;

	//! Handle to the object manager
	private final UAVObjectManager objMngr;

	//! Handle to the apply button
	private Button applyBtn;

	//! Handle to the save button
	private Button saveBtn;

	//! Handle to the load button
	private Button loadBtn;

	//! Indicate if Object Persistence was updated
	private boolean persistenceUpdated;

	//! Indicate if the object was updated
	private boolean objectUpdated;

	//! Handle to the UAVO this class works with
	private final UAVObject obj;

}
