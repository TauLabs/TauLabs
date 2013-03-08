package org.taulabs.androidgcs.views;
/**
 ******************************************************************************
 * @file       ScrollBarView.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      A scrollable view
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

import org.taulabs.androidgcs.R;
import org.taulabs.androidgcs.util.ObjectFieldMappable;

import android.content.Context;
import android.content.res.TypedArray;
import android.text.Editable;
import android.text.InputType;
import android.text.TextWatcher;
import android.util.AttributeSet;
import android.util.Log;
import android.widget.EditText;
import android.widget.GridLayout;
import android.widget.TextView;

public class NumericalFieldView extends GridLayout implements ObjectFieldMappable {

	private final static String TAG = NumericalFieldView.class.getSimpleName();

	private final TextView lbl;
	private final EditText edit;
	private double value;
	private String name;
	private boolean localUpdate = false;

	private Runnable changeListener = null;

	public NumericalFieldView(Context context, AttributeSet attrs) {
		super(context, attrs);

		Log.d(TAG, "NumericalFieldView init called");

		lbl = new TextView(context);
		TypedArray ta = context.obtainStyledAttributes(attrs, R.styleable.setting_attributes, 0, 0);
		lbl.setText(ta.getString(R.styleable.setting_attributes_setting_name));
		addView(lbl, new GridLayout.LayoutParams(spec(0), spec(0)));

		edit = new EditText(context);
		edit.setInputType(InputType.TYPE_NUMBER_FLAG_DECIMAL);
		addView(edit, new GridLayout.LayoutParams(spec(0), spec(1)));

		// Update the value when the edit box changes
		edit.addTextChangedListener(new TextWatcher() {

			@Override
			public void afterTextChanged(Editable s) {
				value = Double.parseDouble(s.toString());
				if (changeListener != null && localUpdate == false)
					changeListener.run();
			}

			@Override
			public void beforeTextChanged(CharSequence s, int start, int count,
					int after) {
			}

			@Override
			public void onTextChanged(CharSequence s, int start, int before,
					int count) {
			}
		});

		setPadding(5,5,5,5);

		setMinimumWidth(300);
		setValue(0);
	}

	public void setName(String n)
	{
		name = n;
		lbl.setText(name);
	}


	@Override
	public double getValue() {
		return value;
	}

	//! Set the input type for the editable field
	public void setInputType(int type) {
		edit.setInputType(type);
	}

	@Override
	public void setValue(double val) {
		localUpdate = true;
		value = val;
		edit.setText(Double.toString(value));
		localUpdate = false;
	}

	@Override
	public void setOnChangedListener(Runnable run) {
		changeListener = run;
	}
}
