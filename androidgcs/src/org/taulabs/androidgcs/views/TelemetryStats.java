/**
 ******************************************************************************
 * @file       TelemetryStats.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      A view to show the telemetry stats
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

import android.content.Context;
import android.graphics.Color;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.widget.LinearLayout;
import android.widget.TextView;

public class TelemetryStats extends LinearLayout {

	public TelemetryStats(Context context) {
		super(context);
		initTelemetryStatsView(context);
	}

	public TelemetryStats(Context context, AttributeSet ats, int defaultStyle) {
		super(context, ats, defaultStyle);
		initTelemetryStatsView(context);
	}

	public TelemetryStats(Context context, AttributeSet ats) {
		super(context, ats);
		initTelemetryStatsView(context);
	}


	public void initTelemetryStatsView(Context context) {
		LayoutInflater inflater = (LayoutInflater) getContext().getSystemService(Context.LAYOUT_INFLATER_SERVICE);
		inflater.inflate(R.layout.telemetry_stats, this);
		setConnected(false);
		invalidate();
	}

	public void setTxRate(int rate) {
		TextView txRate = (TextView) findViewById(R.id.telemetry_stats_tx_rate);
		if (txRate != null)
			txRate.setText(Integer.toString(rate));
		invalidate();
	}

	public void setRxRate(int rate) {
		TextView rxRate = (TextView) findViewById(R.id.telemetry_stats_rx_rate);
		if (rxRate != null)
			rxRate.setText(Integer.toString(rate));
		invalidate();
	}

	public void setConnected(boolean connected) {
		int color = 0x00000000;
		if (connected)
			color = Color.GREEN;
		TextView txRateIcon = (TextView) findViewById(R.id.telemetry_stats_tx_rate_label);
		if (txRateIcon != null) {
			txRateIcon.setBackgroundColor(color);
		}
		TextView rxRateIcon = (TextView) findViewById(R.id.telemetry_stats_rx_rate_label);
		if (rxRateIcon != null) {
			rxRateIcon.setBackgroundColor(color);
		}
	}

}
