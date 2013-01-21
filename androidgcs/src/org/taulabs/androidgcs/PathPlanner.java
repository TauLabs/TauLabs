/**
 ******************************************************************************
 * @file       PathPlanner.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Activity for drawing and modifying paths
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
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.taulabs.androidgcs.R;
import org.taulabs.uavtalk.UAVDataObject;
import org.taulabs.uavtalk.UAVObjectField;
import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;

import android.os.Bundle;
import android.util.Log;
import android.util.Xml;

public class PathPlanner extends ObjectManagerActivity {

	private static final String TAG = PathPlanner.class.getSimpleName();
	private static final boolean DEBUG = true;

	private static final String ns = null;

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.path_planner);
	}

	@Override
	void onOPConnected() {
		super.onOPConnected();

		File waypointsFile = new File("/data/data/waypoints.xml");
		List<Waypoint> waypoints = null;

		try {
			FileInputStream inStream = new FileInputStream(waypointsFile);
			waypoints = parse(inStream);
			if (DEBUG) Log.d(TAG, waypoints.toString());
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (XmlPullParserException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		storeWaypoints(waypoints);
	}

	//! Takes a list of waypoints and copies it to the flash on the UAV
	private void storeWaypoints(List<Waypoint> waypoints) {

		UAVDataObject firstWaypoint = (UAVDataObject) objMngr.getObject("Waypoint");
		if (firstWaypoint == null) {
			Log.e(TAG, "Waypoints are not registered.");
		} else {
			if (waypoints != null && waypoints.size() > 0) {
				for (int i = 0; i < waypoints.size(); i++) {

					Waypoint waypoint = waypoints.get(i);
					UAVDataObject waypointObject = (UAVDataObject) objMngr.getObject("Waypoint",waypoint.instanceId);
					if (waypointObject == null)
						waypointObject = firstWaypoint.clone(waypoint.instanceId);

					UAVObjectField field;

					field = waypointObject.getField("Position");
					field.setValue(waypoint.Position[0], 0);
					field.setValue(waypoint.Position[1], 1);
					field.setValue(waypoint.Position[2], 2);

					field = waypointObject.getField("Velocity");
					field.setValue(waypoint.Velocity[0], 0);
					field.setValue(waypoint.Velocity[1], 1);
					field.setValue(waypoint.Velocity[2], 2);

					field = waypointObject.getField("YawDesired");
					field.setValue(waypoint.YawDesired);

					field = waypointObject.getField("Action");
					field.setValue(waypoint.Action);

					// TODO: Verify all the waypoints are properly and robustly updated
					waypointObject.updated();
				}
			}
		}
	}

	public List<Waypoint> parse(InputStream in) throws XmlPullParserException, IOException {
        try {
            XmlPullParser parser = Xml.newPullParser();
            parser.setFeature(XmlPullParser.FEATURE_PROCESS_NAMESPACES, false);
            parser.setInput(in, null);
            parser.nextTag();
            return readUavobjects(parser);
        } finally {
            in.close();
        }
    }

	//! Start in the uavobjects dump and find the waypoint entry
	private List<Waypoint> readUavobjects(XmlPullParser parser) throws XmlPullParserException, IOException {
	    List <Waypoint>entries = new ArrayList<Waypoint>();

	    parser.require(XmlPullParser.START_TAG, ns, "uavobjects");
	    while (parser.next() != XmlPullParser.END_TAG) {
	        if (parser.getEventType() != XmlPullParser.START_TAG) {
	            continue;
	        }
	        String name = parser.getName();
	        // Starts by looking for the entry tag
	        if (name.equals("waypoints")) {
	        	entries.addAll(readWaypoints(parser));
	        } else {
	            skip(parser);
	        }
	    }

	    return entries;
	}

	//! Go through the waypoints section looking for objects
	private List<Waypoint> readWaypoints(XmlPullParser parser) throws XmlPullParserException, IOException {
	    List <Waypoint>entries = new ArrayList<Waypoint>();

	    parser.require(XmlPullParser.START_TAG, ns, "waypoints");
	    while (parser.next() != XmlPullParser.END_TAG) {
	        if (parser.getEventType() != XmlPullParser.START_TAG) {
	            continue;
	        }
	        String name = parser.getName();
	        // Starts by looking for the entry tag
	        if (name.equals("object")) {
	            entries.add(readWaypoint(parser));
	        } else {
	            skip(parser);
	        }
	    }
	    return entries;
	}

	public static class Waypoint {
		public final int instanceId;
	    public double Position[];
	    public double Velocity[];
	    public final double YawDesired;
	    public String Action;

	    private Waypoint(int instanceId, double [] position, double [] velocity, double yawDesired, String action) {
	        this.instanceId = instanceId;
	        this.Position = position;
	        this.Velocity = velocity;
	        this.YawDesired = yawDesired;
	        this.Action = action;
	    }
	}

	/**
	 * Parses the contents of a waypoint.  Currently it returns a local class that mimics the
	 * structure of the waypoint but this could be made into a more generic method for all
	 * objects and then set directly on the object manager.  However that precludes any sanity
	 * checking before loading the whole path
	 */
	private Waypoint readWaypoint(XmlPullParser parser) throws XmlPullParserException, IOException {
	    parser.require(XmlPullParser.START_TAG, ns, "object");

	    int instanceId = 0;
	    double [] position = new double[3];
	    double [] velocity = new double[3];
	    double yawDesired = 0;
	    String action = "";

	    //TODO: Check name is Waypoint
        instanceId = new Scanner(parser.getAttributeValue(null, "instId")).nextInt();
        String objectName = parser.getAttributeValue(null, "name");
        if (!objectName.equals("Waypoint"))
        	return null;

	    while (parser.next() != XmlPullParser.END_TAG) {
	        if (parser.getEventType() != XmlPullParser.START_TAG) {
	            continue;
	        }

	        String name = parser.getName();
	        if (name.equals("field")) {
	        	String fieldName = parser.getAttributeValue(null, "name");
	        	String value = parser.getAttributeValue(null, "values");
	        	if (fieldName.equals("Position")) {
	        		Scanner s = new Scanner(value).useDelimiter(",");
	        		position[0] = s.nextFloat();
	        		position[1] = s.nextFloat();
	        		position[2] = s.nextFloat();
	        	} else if (fieldName.equals("Velocity")) {
	        		Scanner s = new Scanner(value).useDelimiter(",");
	        		velocity[0] = s.nextFloat();
	        		velocity[1] = s.nextFloat();
	        		velocity[2] = s.nextFloat();
	        	} else if (fieldName.equals("YawDesired")) {
	        		Scanner s = new Scanner(value);
	        		yawDesired = s.nextFloat();
	        	} else if (fieldName.equals("Action")) {
	        		action = value;
	        	}
	        	parser.next();
	        } else {
	            skip(parser);
	        }
	    }
	    if (DEBUG) Log.d(TAG, "instance id: " + instanceId + " " + position[0] + " " + action);

	    return new Waypoint(instanceId, position, velocity, yawDesired, action);
	}

	private void skip(XmlPullParser parser) throws XmlPullParserException, IOException {
	    if (parser.getEventType() != XmlPullParser.START_TAG) {
	        throw new IllegalStateException();
	    }
	    int depth = 1;
	    while (depth != 0) {
	        switch (parser.next()) {
	        case XmlPullParser.END_TAG:
	            depth--;
	            break;
	        case XmlPullParser.START_TAG:
	            depth++;
	            break;
	        }
	    }
	 }

}
