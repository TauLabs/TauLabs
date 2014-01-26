/**
 ******************************************************************************
 * @file       ObjectEditor.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      A popup dialog for editing the contents of a UAVO.
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

import java.util.List;
import java.util.ListIterator;

import org.taulabs.androidgcs.R;
import org.taulabs.androidgcs.util.SmartSave;
import org.taulabs.androidgcs.views.ObjectEditView;
import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectField;
import org.taulabs.uavtalk.UAVObjectManager;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

public class ObjectEditor extends ObjectManagerFragment {

	static final String TAG = "ObjectEditor";
	String objectName;
	long objectID;
	long instID;
	private SmartSave smartSave;
	private boolean updated = false; //!< Indicate if the data has been fetched from UAV

	@Override
	public void setArguments(Bundle b) {
		objectName = b.getString("org.taulabs.androidgcs.ObjectName");
		objectID = b.getLong("org.taulabs.androidgcs.ObjectId");
		instID = b.getLong("org.taulabs.androidgcs.InstId");
		updated = b.getBoolean("org.taulabs.androidgcs.updated", false);
	}


	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
			Bundle savedInstanceState) {
		
		if (savedInstanceState != null) {
			// Unpack the object information
			setArguments(savedInstanceState);
		}

		// Inflate the layout for this fragment
		return inflater.inflate(R.layout.object_editor, container, false);
	}

	private boolean connected = false;
	private boolean resumed = false;
	@Override
	public void onConnected(UAVObjectManager objMngr) {
		super.onConnected(objMngr);
		
		connected = true;
		init();
	}
	
	@Override
	public void onDisconnected() {
		super.onDisconnected();
		
		connected = false;
	}
	
	@Override
	public void onResume() {
		super.onResume();
		
		resumed = true;
		init();
	}

	public void init() {
		// Wait for display to appear and to connect
		if (!connected || !resumed)
			return;

		UAVObject obj = objMngr.getObject(objectID, instID);
		if (obj == null) {
			Log.d(TAG, "Object not found:" + objectID);
			return;
		}

		smartSave = new SmartSave(objMngr, getActivity(),
				obj,
				(Button) getActivity().findViewById(R.id.object_edit_save_button),
				(Button) getActivity().findViewById(R.id.object_edit_apply_button),
				(Button) getActivity().findViewById(R.id.object_edit_load_button));

		ObjectEditView editView = (ObjectEditView) getActivity().findViewById(R.id.object_edit_view);
		editView.setSmartSave(smartSave);
		editView.setName(obj.getName());

		// When a field is added to the edit view then it is linked
		// to the smart save button
		List<UAVObjectField> fields = obj.getFields();
		ListIterator<UAVObjectField> li = fields.listIterator();
		while (li.hasNext()) {
			editView.addField(li.next());
		}
		
		if (!updated) {
			// Only need to update data one time
			smartSave.fetchSettings();
			updated = true;
		}

		smartSave.refreshSettingsDisplay();
	}
	
	@Override
	public void onSaveInstanceState (Bundle outState) {
		super.onSaveInstanceState(outState);
		
		outState.putString("org.taulabs.androidgcs.ObjectName", objectName);
		outState.putLong("org.taulabs.androidgcs.ObjectId", objectID);
		outState.putLong("org.taulabs.androidgcs.InstId", instID);
		outState.putBoolean("org.taulabs.androidgcs.updated", updated);
	}
	
	@Override
	protected String getDebugTag() {
		return TAG;
	}

}
