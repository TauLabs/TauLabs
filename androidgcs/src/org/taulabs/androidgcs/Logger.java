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

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.List;

import org.taulabs.androidgcs.telemetry.OPTelemetryService.TelemTask;
import org.taulabs.androidgcs.telemetry.tasks.LoggingTask;
import org.taulabs.uavtalk.UAVObject;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;


public class Logger extends ObjectManagerActivity {

	final String TAG = "Logger";

	final boolean VERBOSE = false;
	final boolean DEBUG = true;

	private int writtenBytes;
	private int writtenObjects;

	private final List<String> fileList = new ArrayList<String>();
	private File[] fileArray;

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.logger);

		fileList.clear();
		fileArray = getLogFiles();
		if (fileArray != null) {
			for(File file : fileArray)
				fileList.add(file.getName());
		}

		ArrayAdapter<String> logFileListAdapter = new ArrayAdapter<String>(this,
				android.R.layout.simple_list_item_1, fileList);
		getFileListView().setAdapter(logFileListAdapter);

		getFileListView().setOnItemClickListener(new AdapterView.OnItemClickListener() {
		    @Override
			public void onItemClick(AdapterView<?> parent, View v, int position, long id) {
		    	Log.d(TAG, fileArray[position].getAbsolutePath());

		    	Intent intent = new Intent(Intent.ACTION_SEND);
	               intent.setType("application/octet-stream");
	               intent.putExtra(Intent.EXTRA_EMAIL, "noreply@taulabs");
	               intent.putExtra(Intent.EXTRA_SUBJECT, "Tau Labs log file");
	               intent.putExtra(Intent.EXTRA_TEXT, fileArray[position].getName());
	               intent.putExtra(Intent.EXTRA_STREAM, Uri.parse("file://"+ fileArray[position].getAbsolutePath()));
	               startActivity(Intent.createChooser(intent, "Choice app to send file:"));
		    }
		});

	}

	//! Return the file list view
	private ListView getFileListView() {
		return (ListView) findViewById(R.id.logger_file_list);
	}

	@Override
	void onOPConnected() {
		super.onOPConnected();

		UAVObject stats = objMngr.getObject("FlightTelemetryStats");
		if (stats != null) {
			registerObjectUpdates(stats);
		}
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

	private File[] getLogFiles() {
		File root = Environment.getExternalStorageDirectory();
		File logDirectory = new File(root, "/TauLabs");
		return logDirectory.listFiles(new FilenameFilter() {

			@Override
			public boolean accept(File dir, String filename) {
				return filename.contains(".opl");
			}
		});
	}


}
