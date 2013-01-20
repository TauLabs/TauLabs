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
import android.text.TextWatcher;
import android.util.AttributeSet;
import android.util.Log;
import android.widget.EditText;
import android.widget.GridLayout;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.SeekBar.OnSeekBarChangeListener;
import android.widget.TextView;

public class ScrollBarView extends GridLayout implements ObjectFieldMappable {

	private final static String TAG = ScrollBarView.class.getSimpleName();

	private final TextView lbl;
	private final EditText edit;
	private final SeekBar bar;
	private double value;
	private String name;
	private boolean localUpdate = false;

	private final double SCALE = 1000000;
	private Runnable changeListener = null;

	public ScrollBarView(Context context, AttributeSet attrs) {
		super(context, attrs);

		Log.d(TAG, "Scroll bar init called");

		setOrientation(LinearLayout.VERTICAL);
		setColumnCount(2);

		lbl = new TextView(context);
		TypedArray ta = context.obtainStyledAttributes(attrs, R.styleable.setting_attributes, 0, 0);
		lbl.setText(ta.getString(R.styleable.setting_attributes_setting_name));
		addView(lbl, new GridLayout.LayoutParams(spec(0), spec(0)));

		edit = new EditText(context);
		addView(edit, new GridLayout.LayoutParams(spec(0), spec(1)));

		bar = new SeekBar(context);
		addView(bar, new GridLayout.LayoutParams(spec(1), spec(0,2)));

		ta = context.obtainStyledAttributes(attrs, R.styleable.setting_attributes, 0, 0);
		final double max = ta.getFloat(R.styleable.setting_attributes_max_value,0);
		bar.setMax((int) (SCALE * max));

		// Update the value when the progress bar changes
		bar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress,
					boolean fromUser) {
				value = progress / SCALE;
				edit.setText(Double.toString(value));
				if (changeListener != null && localUpdate == false)
					changeListener.run();
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}
		});

		// Update the value when the edit box changes
		edit.addTextChangedListener(new TextWatcher() {

			@Override
			public void afterTextChanged(Editable s) {
				value = Double.parseDouble(s.toString());
				bar.setProgress((int) (SCALE * value));
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
		setValue(0.0035);
	}

	public void setName(String n)
	{
		name = n;
		lbl.setText(name);
	}

	@Override
	protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
		super.onMeasure(widthMeasureSpec, heightMeasureSpec);

		// This shouldn't be needed if I could make this scroll bar
		// automagically span both columns
		android.view.ViewGroup.LayoutParams param = bar.getLayoutParams();
		param.width = (int) (getMeasuredWidth() * 0.9);

		// Force the label to half the page width
		param = lbl.getLayoutParams();
		param.width = getMeasuredWidth() / 2;
	}

	@Override
	public double getValue() {
		return value;
	}

	@Override
	public void setValue(double val) {
		localUpdate = true;
		value = val;
		edit.setText(Double.toString(value));
		bar.setProgress((int) (SCALE * value));
		localUpdate = false;
	}

	@Override
	public void setOnChangedListener(Runnable run) {
		changeListener = run;
	}
}
