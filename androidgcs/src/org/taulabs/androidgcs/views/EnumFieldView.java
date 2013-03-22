package org.taulabs.androidgcs.views;
/**
 ******************************************************************************
 * @file       EnumFieldView.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Display and edit enum fields from UAVOs
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

import java.util.List;

import org.taulabs.androidgcs.util.ObjectFieldMappable;
import org.taulabs.uavtalk.UAVObjectField;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.ArrayAdapter;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;

public class EnumFieldView extends LinearLayout implements ObjectFieldMappable {

	private final static String TAG = EnumFieldView.class.getSimpleName();

	private final TextView lbl;
	private final Spinner  spin;
	private double value;
	private boolean localUpdate = false;

	private Runnable changeListener = null;

	//! This is the constructor used by the SDK for setting it up
	public EnumFieldView(Context context, AttributeSet attrs) {
		super(context, attrs);

		final int WIDTH = (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 200, getResources().getDisplayMetrics());

		lbl = new TextView(context);
		lbl.setText("Field: ");
		addView(lbl, new LinearLayout.LayoutParams(WIDTH, LayoutParams.WRAP_CONTENT, 1));

		spin = new Spinner(context);
		addView(spin, new LinearLayout.LayoutParams(WIDTH, LayoutParams.WRAP_CONTENT, 1));

		// Update the value when the edit box changes
		spin.setOnItemSelectedListener(new OnItemSelectedListener() {

			@Override
			public void onItemSelected(AdapterView<?> arg0, View arg1,
					int arg2, long arg3) {
				value = spin.getSelectedItemPosition();
				if (changeListener != null && localUpdate == false)
					changeListener.run();
			}

			@Override
			public void onNothingSelected(AdapterView<?> arg0) {
			}
		});

		setPadding(5,5,5,5);

		setMinimumWidth(300);
		setValue(0);
	}

	//! This is the constructor used by the code
	public EnumFieldView(Context context, AttributeSet attrs, UAVObjectField field, int idx) {
		this(context, attrs);

		// Set the label name
		String name = field.getName();
		List<String> elements = field.getElementNames();
		if (elements != null && elements.size() > 1) {
			name = name + "-" + elements.get(idx);
		}
		lbl.setText(name);

		ArrayAdapter<String> adapter = new ArrayAdapter<String>(context, android.R.layout.simple_spinner_item);
		adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
		adapter.addAll(field.getOptions());
		spin.setAdapter(adapter);
		spin.setSelection((int) field.getDouble(idx));
	}

	@Override
	public double getValue() {
		return value;
	}

	@Override
	public void setValue(double val) {
		localUpdate = true;
		Log.d(TAG, "Value set to: " + val);
		value = val;
		spin.setSelection((int) val);
		localUpdate = false;
	}

	@Override
	public void setOnChangedListener(Runnable run) {
		changeListener = run;
	}
}
