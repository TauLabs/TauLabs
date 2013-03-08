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
package org.taulabs.androidgcs;

import java.util.List;
import java.util.ListIterator;

import org.taulabs.androidgcs.util.SmartSave;
import org.taulabs.uavtalk.UAVObject;
import org.taulabs.uavtalk.UAVObjectField;

import android.os.Bundle;
import android.util.Log;
import android.widget.Button;

public class ObjectEditor extends ObjectManagerActivity {

	static final String TAG = "ObjectEditor";
	String objectName;
	long objectID;
	long instID;
	private SmartSave smartSave;


	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.object_editor);

		// TODO: Figure out why this line is required so it doesn't
		// have to be set programmatically
		setTheme(android.R.style.Theme_Holo);

		Bundle extras = getIntent().getExtras();
		if (extras == null)
			return;

		objectName = extras.getString("org.taulabs.androidgcs.ObjectName");
		objectID = extras.getLong("org.taulabs.androidgcs.ObjectId");
		instID = extras.getLong("org.taulabs.androidgcs.InstId");

		setTitle(objectName);

	}

	@Override
	public void onOPConnected() {
		UAVObject obj = objMngr.getObject(objectID, instID);
		if (obj == null) {
			Log.d(TAG, "Object not found:" + objectID);
			return;
		}

		smartSave = new SmartSave(objMngr, this,
				obj,
				(Button) findViewById(R.id.object_edit_save_button),
				(Button) findViewById(R.id.object_edit_apply_button),
				(Button) findViewById(R.id.object_edit_load_button));

		ObjectEditView editView = (ObjectEditView) findViewById(R.id.object_edit_view);
		editView.setSmartSave(smartSave);
		editView.setName(obj.getName());

		// When a field is added to the edit view then it is linked
		// to the smart save button
		List<UAVObjectField> fields = obj.getFields();
		ListIterator<UAVObjectField> li = fields.listIterator();
		while (li.hasNext()) {
			editView.addField(li.next());
		}
		smartSave.refreshSettingsDisplay();
	}
}
