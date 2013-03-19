/**
 ******************************************************************************
 * @file       TcpUAVTalk.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      UAVTalk over TCP.
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
package org.taulabs.androidgcs.telemetry;

import android.content.Context;
import android.hardware.usb.UsbManager;
import android.util.Log;

import com.hoho.android.usbserial.driver.UsbSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialProber;

public class SerialUAVTalk extends TelemetryTask {
	private final String TAG = SerialUAVTalk.class.getSimpleName();
	public static int LOGLEVEL = 2;
	public static boolean WARN = LOGLEVEL > 1;
	public static boolean DEBUG = LOGLEVEL > 0;

	private UsbManager mUsbManager;
	private UsbSerialDriver mSerialDevice;
	private final UsbSerialDriver driver = null;

	/**
	 * Construct a TcpUAVTalk object attached to the OPTelemetryService.  Gets the
	 * connection settings from the preferences.
	 */
	public SerialUAVTalk(OPTelemetryService caller) {
		super(caller);
	}

	@Override
	boolean attemptConnection() {

		if( getConnected() )
			return true;

		UsbSerialProber.setConnectedRunnable(new Runnable() {
			@Override
			public void run() {
				Log.d(TAG, "Connected");
				mSerialDevice = UsbSerialProber.driver;
				attemptSucceeded();
			}
		}, handler);

		mUsbManager = (UsbManager) telemService.getSystemService(Context.USB_SERVICE);
		return UsbSerialProber.acquire(mUsbManager, telemService);
	}


	@Override
	public void disconnect() {
		super.disconnect();
	}

}
