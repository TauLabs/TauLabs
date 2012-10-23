package org.openpilot.androidgcs.telemetry.tasks;

import org.openpilot.uavtalk.UAVObjectManager;

public interface ITelemTask {
	public void connect(UAVObjectManager objMngr);
	public void disconnect();
}
