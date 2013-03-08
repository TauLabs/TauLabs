/**
 ******************************************************************************
 * @file       ObjectEditView.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      The view for editing a UAVO
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

import java.util.ArrayList;
import java.util.List;

import org.taulabs.androidgcs.util.SmartSave;
import org.taulabs.androidgcs.views.EnumFieldView;
import org.taulabs.androidgcs.views.NumericalFieldView;
import org.taulabs.uavtalk.UAVObjectField;

import android.content.Context;
import android.text.InputType;
import android.util.AttributeSet;
import android.view.View;
import android.widget.EditText;
import android.widget.GridLayout;
import android.widget.LinearLayout;

public class ObjectEditView extends GridLayout {

	String objectName;
	public List<View> fields;
	private SmartSave smartSave;

	public ObjectEditView(Context context) {
		super(context);
		initObjectEditView();
	}

	public ObjectEditView(Context context, AttributeSet ats, int defaultStyle) {
		super(context, ats);
		initObjectEditView();
	}

	public ObjectEditView(Context context, AttributeSet ats) {
		super(context, ats);
		initObjectEditView();
	}

	public void setSmartSave(SmartSave s) {
		smartSave = s;
	}

	public void initObjectEditView() {
		// Set orientation of layout to vertical
		setOrientation(LinearLayout.VERTICAL);
		setColumnCount(2);
		fields = new ArrayList<View>();
	}

	public void setName(String name) {
		objectName = name;
	}

	public void addField(UAVObjectField field) {
		for (int i = 0; i < field.getNumElements(); i++)
			addRow(getContext(), field, i);
	}


	public void addRow(Context context, UAVObjectField field, int idx) {
		int row = getRowCount();

		/*TextView fieldName = new TextView(context);
		if(field.getNumElements() == 1) {
			fieldName.setText(field.getName());
		} else {
			fieldName.setText(field.getName() + "-" + field.getElementNames().get(idx));
		}
		addView(fieldName, new GridLayout.LayoutParams(spec(row), spec(0))); */

		View fieldValue = null;
		switch(field.getType())
		{
		case FLOAT32:
			fieldValue = new NumericalFieldView(context, null, field, idx);
			((NumericalFieldView)fieldValue).setValue(Double.parseDouble(field.getValue(idx).toString()));
			((NumericalFieldView)fieldValue).setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_SIGNED | InputType.TYPE_NUMBER_FLAG_DECIMAL);
			if (smartSave != null)
				smartSave.addControlMapping((NumericalFieldView)fieldValue, field.getName(), idx);
			break;
		case INT8:
		case INT16:
		case INT32:
			fieldValue = new NumericalFieldView(context, null, field, idx);
			((NumericalFieldView)fieldValue).setValue(Double.parseDouble(field.getValue(idx).toString()));
			((NumericalFieldView)fieldValue).setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_SIGNED);
			if (smartSave != null)
				smartSave.addControlMapping((NumericalFieldView)fieldValue, field.getName(), idx);
			break;
		case UINT8:
		case UINT16:
		case UINT32:
			fieldValue = new NumericalFieldView(context, null, field, idx);
			((NumericalFieldView)fieldValue).setValue(Double.parseDouble(field.getValue(idx).toString()));
			((NumericalFieldView)fieldValue).setInputType(InputType.TYPE_CLASS_NUMBER);
			if (smartSave != null)
				smartSave.addControlMapping((NumericalFieldView)fieldValue, field.getName(), idx);
			break;
		case ENUM:
			fieldValue = new EnumFieldView(context, null, field, idx);
			if (smartSave != null)
				smartSave.addControlMapping((EnumFieldView)fieldValue, field.getName(), idx);
			break;
		case BITFIELD:
			fieldValue = new EditText(context);
			((EditText)fieldValue).setText(field.getValue(idx).toString());
			((EditText)fieldValue).setInputType(InputType.TYPE_CLASS_NUMBER);
			break;
		case STRING:
			fieldValue = new EditText(context);
			((EditText)fieldValue).setText(field.getValue(idx).toString());
		}

		addView(fieldValue, new GridLayout.LayoutParams(spec(row), spec(0)));
		fields.add(fieldValue);
	}

}
