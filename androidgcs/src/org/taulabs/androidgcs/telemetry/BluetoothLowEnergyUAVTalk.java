/**
 ******************************************************************************
 * @file       BluetoothUAVTalk.java
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      Telemetry over bluetooth.
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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;

import com.polkapolka.bluetooth.le.BluetoothLeService;
import com.polkapolka.bluetooth.le.SampleGattAttributes;

import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattService;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.util.Log;

public class BluetoothLowEnergyUAVTalk extends TelemetryTask {

	private final String TAG = BluetoothLowEnergyUAVTalk.class.getSimpleName();
	public static final int LOGLEVEL = 4;
	public static final boolean DEBUG = LOGLEVEL > 2;
	public static final boolean WARN = LOGLEVEL > 1;
	public static final boolean ERROR = LOGLEVEL > 0;
	
    public static final String EXTRAS_DEVICE_NAME = "DEVICE_NAME";
    public static final String EXTRAS_DEVICE_ADDRESS = "DEVICE_ADDRESS";

    private String mDeviceName = "HMSoft";
    private String mDeviceAddress = "78:A5:04:3E:D6:08";
    private BluetoothLeService mBluetoothLeService;
    private boolean mConnected = false;
    private BluetoothGattCharacteristic characteristicTX;
    private BluetoothGattCharacteristic characteristicRX;

	//! The stream that receives data from the HID device
	private TalkInputStream inTalkStream;
	//! The stream which sends data to the HID device
	private TalkOutputStream outTalkStream;

    public final static UUID HM_RX_TX =
            UUID.fromString(SampleGattAttributes.HM_RX_TX);

    private final String LIST_NAME = "NAME";
    private final String LIST_UUID = "UUID";
    
	public BluetoothLowEnergyUAVTalk(TelemetryService caller) {
		super(caller);
	}

	@Override
	boolean attemptConnection() {
		// Connect to the bluetooth low energy service

		if (DEBUG) Log.d(TAG, "Attempting to connect to BT Le service");

		telemService.registerReceiver(mGattUpdateReceiver, makeGattUpdateIntentFilter());
		
		// Once the BT Le service connects it will automatically attempt to open device
		Intent gattServiceIntent = new Intent(telemService, BluetoothLeService.class);
        telemService.bindService(gattServiceIntent, mServiceConnection, Context.BIND_AUTO_CREATE);

		if( getConnected() )
			return true;
		
		return true;
	}

	@Override
	public void disconnect() {
		super.disconnect();
		mBluetoothLeService.disconnect();
		telemService.unregisterReceiver(mGattUpdateReceiver);
		telemService.unbindService(mServiceConnection);
        mBluetoothLeService = null;
	}
	
	@Override
	public boolean getConnected() {
		return mConnected;
	}

    // Code to manage Service lifecycle.
    private final ServiceConnection mServiceConnection = new ServiceConnection() {

        @Override
        public void onServiceConnected(ComponentName componentName, IBinder service) {
        	if (DEBUG) Log.d(TAG, "Service connected. Attempting to open device: " + mDeviceAddress);
            mBluetoothLeService = ((BluetoothLeService.LocalBinder) service).getService();
            if (!mBluetoothLeService.initialize()) {
                Log.e(TAG, "Unable to initialize Bluetooth");
                // TODO: throw failure code
            }
            // Automatically connects to the device upon successful start-up initialization.
            mBluetoothLeService.connect(mDeviceAddress);
        }

        @Override
        public void onServiceDisconnected(ComponentName componentName) {
            mBluetoothLeService = null;
        }
    };

    // Handles various events fired by the Service.
    // ACTION_GATT_CONNECTED: connected to a GATT server.
    // ACTION_GATT_DISCONNECTED: disconnected from a GATT server.
    // ACTION_GATT_SERVICES_DISCOVERED: discovered GATT services.
    // ACTION_DATA_AVAILABLE: received data from the device.  This can be a result of read
    //                        or notification operations.
    private final BroadcastReceiver mGattUpdateReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            final String action = intent.getAction();
            if (BluetoothLeService.ACTION_GATT_CONNECTED.equals(action)) {
                mConnected = true;
                
                if (DEBUG) Log.d(TAG, "Device connected");

        		telemService.toastMessage("Bluetooth device connected");

        		inTalkStream = new TalkInputStream();
        		outTalkStream = new TalkOutputStream();
        		inStream = inTalkStream;
        		outStream = outTalkStream;

        		// Post message to call attempt succeeded on the parent class
        		handler.post(new Runnable() {
        			@Override
        			public void run() {
        				BluetoothLowEnergyUAVTalk.this.attemptSucceeded();
        			}
        		});
        		        		
        		// TODO: create inStream and outStream here
        		
                Log.d(TAG, "ACTION_GATT_CONNECTED");
            } else if (BluetoothLeService.ACTION_GATT_DISCONNECTED.equals(action)) {
                mConnected = false;
                
                telemService.toastMessage("Bluetooth device disconnected");

                Log.d(TAG, "ACTION_GATT_DISCONNECTED");
            } else if (BluetoothLeService.ACTION_GATT_SERVICES_DISCOVERED.equals(action)) {
                // Show all the supported services and characteristics on the user interface.
                displayGattServices(mBluetoothLeService.getSupportedGattServices());
            } else if (BluetoothLeService.ACTION_DATA_AVAILABLE.equals(action)) {
            	Log.d(TAG, "Data available");
            	receiveData(intent.getStringExtra(BluetoothLeService.EXTRA_DATA));
            }
        }
    };

 
    // Demonstrates how to iterate through the supported GATT Services/Characteristics.
    // In this sample, we populate the data structure that is bound to the ExpandableListView
    // on the UI.
    private void displayGattServices(List<BluetoothGattService> gattServices) {
    	
    	if (DEBUG) Log.d(TAG, "displayGattServices");
    	
        if (gattServices == null) return;
        String uuid = null;
        String unknownServiceString = "Unknown service";
        ArrayList<HashMap<String, String>> gattServiceData = new ArrayList<HashMap<String, String>>();

 
        // Loops through available GATT Services.
        for (BluetoothGattService gattService : gattServices) {
            HashMap<String, String> currentServiceData = new HashMap<String, String>();
            uuid = gattService.getUuid().toString();
            currentServiceData.put(
                    LIST_NAME, SampleGattAttributes.lookup(uuid, unknownServiceString));
            
            // If the service exists for HM 10 Serial, say so.
            if(SampleGattAttributes.lookup(uuid, unknownServiceString) == "HM 10 Serial") { 
            	if (DEBUG) Log.d(TAG ,"Found serial service");
        	} else {  
        		Log.e(TAG, "Unable to find the serial service for BT device");
    		} 
            currentServiceData.put(LIST_UUID, uuid);
            gattServiceData.add(currentServiceData);

     		// get characteristic when UUID matches RX/TX UUID
            characteristicTX = gattService.getCharacteristic(BluetoothLeService.UUID_HM_RX_TX);
            characteristicRX = gattService.getCharacteristic(BluetoothLeService.UUID_HM_RX_TX);
        }
        
    }

    private static IntentFilter makeGattUpdateIntentFilter() {
        final IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(BluetoothLeService.ACTION_GATT_CONNECTED);
        intentFilter.addAction(BluetoothLeService.ACTION_GATT_DISCONNECTED);
        intentFilter.addAction(BluetoothLeService.ACTION_GATT_SERVICES_DISCOVERED);
        intentFilter.addAction(BluetoothLeService.ACTION_DATA_AVAILABLE);
        return intentFilter;
    }
    

    private void writeData(byte[] tx) {
		 if(mConnected && characteristicTX != null) {
		    characteristicTX.setValue(tx);
			mBluetoothLeService.writeCharacteristic(characteristicTX);
			mBluetoothLeService.setCharacteristicNotification(characteristicRX,true);
		 }
    }
    
    private void receiveData(String data) {
    	if (DEBUG) Log.d(TAG, "Received data: " + data);
    	inTalkStream.write(data.getBytes());
    }

	/*********** Helper classes for telemetry streams ************/

	class TalkOutputStream extends OutputStream {

		@Override
		public void write(int oneByte) throws IOException {
			Log.d(TAG, "Writing byte");
			byte [] b = new byte[1];
			b[0] = (byte) oneByte;
			writeData(b);
		}
		
		@Override
		public void write(byte [] bytes) throws IOException {
			
			final int STRIDE = 20; // max size of BT device
			final int LEN = bytes.length;
			int idx = 0;
			
			while(idx < LEN) {
				int n = LEN - idx;
				if (n > STRIDE)
					n = STRIDE;
				byte [] send = new byte[n];
				for (int i = 0; i < n; i++)
					send[i] = bytes[idx + i];

				Log.d(TAG, "Sending " + n + " bytes starting at " + idx + " out of " + LEN);

				writeData(send);
				idx = idx + n;
			}
		}
	};

	private class TalkInputStream extends InputStream {
		// Uses ByteFifo.getByteBlocking()
		// Uses ByteFifo.put(byte[])
		ByteFifo data = new ByteFifo();

		@Override
		public int read() {
			try {
				return data.getByteBlocking();
			} catch (InterruptedException e) {
				if (!shutdown) {
					Log.e(TAG, "Timed out");
					if (DEBUG) e.printStackTrace();
					disconnect();
					telemService.connectionBroken();
				}
			}
			return -1;
		}

		public void write(byte[] b) {
			data.put(b);
		}
	};

	private class ByteFifo {

		//! The maximum size of the fifo
		private final int MAX_SIZE = 256;
		//! The number of bytes in the buffer
		private int size = 0;
		//! Internal buffer
		private final ByteBuffer buf;

		ByteFifo() {
			buf = ByteBuffer.allocate(MAX_SIZE);
			size = 0;
		}

		private int byteToInt(byte b) { return b & 0x000000ff; }

		final int remaining() { return size; };

		public boolean put(byte b) {
			byte[] a = {b};
			return put(a);
		}

		public boolean put(byte[] dat) {
			if ((size + dat.length) > MAX_SIZE) {
				Log.e(TAG, "Dropped data.  Size:" + size + " data length: " + dat.length);
				return false;
			}

			// Place data at the end of the buffer
			synchronized(buf) {
				buf.position(size);
				buf.put(dat);
				size = size + dat.length;
				buf.notify();
			}
			return true;
		}

		//! Return buffer if enough bytes are available
		public ByteBuffer get(byte[] dst, int offset, int size) {
			if (size < this.size)
				return null;

			synchronized(buf) {
				buf.flip();
				buf.get(dst, offset, size);
				buf.compact();
				this.size -= size;
			}
			return buf;
		}

		public int getByteBlocking() throws InterruptedException {
			synchronized(buf) {
				while (size <= 0) {
					buf.wait();
				}
				int val = byteToInt(buf.get(0));
				buf.position(1);
				buf.compact();
				size--;
				return val;
			}
		}
	}
}
