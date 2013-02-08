/**
 ******************************************************************************
 * @file       AlarmsSummary.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      A view to show the current alarm status
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
package org.taulabs.androidgcs.views;

import org.taulabs.androidgcs.R;
import org.taulabs.androidgcs.SystemAlarmActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

public class AlarmsSummary extends LinearLayout {

	public enum AlarmsStatus {
		GOOD,
		WARNING,
		ERROR,
		CRITICAL
	};

	public AlarmsSummary(Context context) {
		super(context);
		initAlarmsSummaryView(context);
	}

	public AlarmsSummary(Context context, AttributeSet ats, int defaultStyle) {
		super(context, ats, defaultStyle);
		initAlarmsSummaryView(context);
	}

	public AlarmsSummary(Context context, AttributeSet ats) {
		super(context, ats);
		initAlarmsSummaryView(context);
	}


	public void initAlarmsSummaryView(Context context) {
		LayoutInflater inflater = (LayoutInflater) getContext().getSystemService(Context.LAYOUT_INFLATER_SERVICE);
		inflater.inflate(R.layout.alarms_summary, this);
		setAlarmsStatus(AlarmsStatus.GOOD);
		TextView alarmsIcon = (TextView) findViewById(R.id.alarms_icon);
		alarmsIcon.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {
				Activity activity = ((Activity)getContext());
				String currentActivity = activity.getClass().getSimpleName();

				// Do not nest multiple of the alarm activities
				if(!currentActivity.equals("SystemAlarmActivity"))
					activity.startActivity(new Intent(getContext(), SystemAlarmActivity.class));
			}

		});
		invalidate();
	}



	//! Set the icon based on the alarm level
	public void setAlarmsStatus(AlarmsStatus status) {
		TextView alarmsIcon = (TextView) findViewById(R.id.alarms_icon);
		Drawable img = null;
		switch (status) {
		case GOOD:
			img = getContext().getResources().getDrawable( R.drawable.ic_alarms_good);
			break;
		case WARNING:
			img = getContext().getResources().getDrawable( R.drawable.ic_alarms_warning);
			break;
		case ERROR:
			img = getContext().getResources().getDrawable( R.drawable.ic_alarms_error);
			break;
		case CRITICAL:
			img = getContext().getResources().getDrawable( R.drawable.ic_alarms_critical);
			break;
		}
		if (alarmsIcon != null && img != null)
			alarmsIcon.setCompoundDrawablesWithIntrinsicBounds( img, null, null, null );
	}
}
