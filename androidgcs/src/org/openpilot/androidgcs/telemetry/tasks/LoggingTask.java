package org.openpilot.androidgcs.telemetry.tasks;

import java.util.ArrayList;
import java.util.List;
import java.util.Observable;
import java.util.Observer;

import org.openpilot.uavtalk.UAVObject;
import org.openpilot.uavtalk.UAVObjectManager;

public class LoggingTask implements ITelemTask {
	private UAVObjectManager objMngr;
	private final List<UAVObject> listeningList = new ArrayList<UAVObject>();
	private boolean loggingActive = false;

	@Override
	public void connect(UAVObjectManager o) {
		objMngr = o;

		// When new objects are registered, ensure we listen
		// to them
		o.addNewObjectObserver(newObjObserver);
		o.addNewInstanceObserver(newObjObserver);

		List<List<UAVObject>> objects = objMngr.getObjects();
		for(int i = 0; i < objects.size(); i++)
			for(int j = 0; j < objects.get(i).size(); j++)
				registerObject(objects.get(i).get(j));

		startLogging();
	}

	@Override
	public void disconnect() {
		endLogging();
		objMngr.deleteNewObjectObserver(newObjObserver);
		objMngr.deleteNewInstanceObserver(newObjObserver);
		unregisterAllObjects();
	}

	//! Register an object to inform this task on updates for logging
	private void registerObject(UAVObject obj) {
		synchronized(listeningList) {
			if (!listeningList.contains(obj)) {
				obj.addUpdatedObserver(objUpdatedObserver);
				listeningList.add(obj);
			}
		}
	}

	//! Unregister all objects from logging
	private void unregisterAllObjects() {
		synchronized(listeningList) {
			for (int i = 0; i < listeningList.size(); i++) {
				listeningList.get(i).removeUpdatedObserver(objUpdatedObserver);
			}
			listeningList.clear();
		}
	}


	//! Write an updated object to the log file
	private void logObject(UAVObject obj) {
		if (loggingActive) {

		}
	}

	//! Open a file and start logging
	private boolean startLogging() {
		loggingActive = true;

		return loggingActive;
	}

	//! Close the file and end logging
	private boolean endLogging() {
		loggingActive = false;

		return true;
	}

	//! Observer to catch when new objects or instances are registered
	private final Observer newObjObserver = new Observer() {
		@Override
		public void update(Observable observable, Object data) {
			UAVObject obj = (UAVObject) data;
			registerObject(obj);
		}
	};

	//! Observer to catch when objects are updated
	private final Observer objUpdatedObserver = new Observer() {
		@Override
		public void update(Observable observable, Object data) {
			UAVObject obj = (UAVObject) data;
			logObject(obj);
		}
	};
}
