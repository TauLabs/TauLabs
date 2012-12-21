/**
 ******************************************************************************
 * @file       Logger.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Controller for logging data as well as interface for getting that
 *             data on and off the tablet.
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

import org.taulabs.androidgcs.telemetry.OPTelemetryService.TelemTask;
import org.taulabs.androidgcs.telemetry.tasks.LoggingTask;
import org.taulabs.uavtalk.UAVObject;

import android.os.Bundle;
import android.widget.TextView;


public class Logger extends ObjectManagerActivity {

	final String TAG = "Logger";

	final boolean VERBOSE = false;
	final boolean DEBUG = true;

	private int writtenBytes;
	private int writtenObjects;

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.logger);
	}

	@Override
	void onOPConnected() {
		super.onOPConnected();

		UAVObject stats = objMngr.getObject("FlightTelemetryStats");
		registerObjectUpdates(stats);
	}

	@Override
	public void objectUpdated(UAVObject obj) {
		TelemTask task = binder.getTelemTask(0);
		LoggingTask logger = task.getLoggingTask();
		writtenBytes = logger.getWrittenBytes();
		writtenObjects = logger.getWrittenObjects();
		((TextView) findViewById(R.id.logger_number_of_bytes)).setText(Integer.valueOf(writtenBytes).toString());
		((TextView) findViewById(R.id.logger_number_of_objects)).setText(Integer.valueOf(writtenObjects).toString());
	}


}
