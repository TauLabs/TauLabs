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

import com.polkapolka.bluetooth.le.SampleGattAttributes;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothProfile;
import android.content.Context;
import android.util.Log;

public class BluetoothLowEnergyUAVTalk extends TelemetryTask {

	private final String TAG = BluetoothLowEnergyUAVTalk.class.getSimpleName();
	public static final int LOGLEVEL = 1;
	public static final boolean VERBOSE = LOGLEVEL > 3;
	public static final boolean DEBUG = LOGLEVEL > 2;
	public static final boolean WARN = LOGLEVEL > 1;
	public static final boolean ERROR = LOGLEVEL > 0;
	
    public static final String EXTRAS_DEVICE_NAME = "DEVICE_NAME";
    public static final String EXTRAS_DEVICE_ADDRESS = "DEVICE_ADDRESS";

    private String mDeviceName = "HMSoft";
    private String mDeviceAddress = "78:A5:04:3E:D6:08";
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

		if( getConnected() )
			return true;
		
		if (DEBUG) Log.d(TAG, "Attempting to connect to BT Le service");

		initialize();
		btleConnect(mDeviceAddress);
		
		return true;
	}

	@Override
	public void disconnect() {
		super.disconnect();
		btleDisconnect();
		close();
	}
	
	@Override
	public boolean getConnected() {
		return mConnected;
	}
 
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
        		if (ERROR) Log.e(TAG, "Unable to find the serial service for BT device");
    		} 
            currentServiceData.put(LIST_UUID, uuid);
            gattServiceData.add(currentServiceData);

     		// get characteristic when UUID matches RX/TX UUID
            characteristicTX = gattService.getCharacteristic(UUID_HM_RX_TX);
            characteristicRX = gattService.getCharacteristic(UUID_HM_RX_TX);
            if (characteristicRX != null)
                setCharacteristicNotification(characteristicRX,true);
        }
        
    }

    private void writeData(byte[] tx) {
		 if(mConnected && characteristicTX != null) {
			if (DEBUG) Log.d(TAG, "Sending data: " + tx + " # bytes: " + tx.length);

		    characteristicTX.setValue(tx);
			writeCharacteristic(characteristicTX);
		 } else {
			 if (ERROR) Log.e(TAG, "Invalid connection: " + mConnected + " " + characteristicTX);
		 }
    }
    
    private void receiveData(String data) {
    	//if (DEBUG) Log.d(TAG, "Received data: " + data);
    	
    	inTalkStream.write(data.getBytes());
    	inTalkStream.notify();
    }
    
    /*********** Helper classes for bluetooth handling ***********/
    
    public final static String EXTRA_DATA =
            "com.example.bluetooth.le.EXTRA_DATA";
    public final static UUID UUID_HM_RX_TX =
            UUID.fromString(SampleGattAttributes.HM_RX_TX);
    
    private BluetoothManager mBluetoothManager;
    private BluetoothAdapter mBluetoothAdapter;
    private String mBluetoothDeviceAddress;
    private BluetoothGatt mBluetoothGatt;
    private int mConnectionState = STATE_DISCONNECTED;
    private static final int STATE_DISCONNECTED = 0;
    private static final int STATE_CONNECTING = 1;
    private static final int STATE_CONNECTED = 2;

    // Implements callback methods for GATT events that the app cares about.  For example,
    // connection change and services discovered.
    private final BluetoothGattCallback mGattCallback = new BluetoothGattCallback() {
        @Override
        public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {

                if (DEBUG) Log.d(TAG, "Device connected");
                
                // Attempts to discover services after successful connection.
                mBluetoothGatt.discoverServices();

            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                mConnected = false;
                telemService.toastMessage("Bluetooth device disconnected");
                if (DEBUG) Log.d(TAG, "ACTION_GATT_DISCONNECTED");
            }
        }

        @Override
        public void onServicesDiscovered(BluetoothGatt gatt, int status) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
            	displayGattServices(getSupportedGattServices());

        		telemService.toastMessage("Bluetooth device connected");

            	mConnected = true;
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
        		
            } else {
            	if (WARN) Log.w(TAG, "onServicesDiscovered received: " + status);
            }
        }

        @Override
        public void onCharacteristicRead(BluetoothGatt gatt,
                                         BluetoothGattCharacteristic characteristic,
                                         int status) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
            	if (DEBUG) Log.d(TAG, "onCharacteristicRead: " + characteristic.toString());
                // TODO: broadcastUpdate(ACTION_DATA_AVAILABLE, characteristic);
            }
        }

        @Override
        public void onCharacteristicChanged(BluetoothGatt gatt,
                                            BluetoothGattCharacteristic characteristic) {
        	if (DEBUG) Log.d(TAG, "onCharacteristicChanged: " + characteristic.toString());
        	final byte[] data = characteristic.getValue();
        	inTalkStream.write(data);
        }
    };


    /**
     * Initializes a reference to the local Bluetooth adapter.
     *
     * @return Return true if the initialization is successful.
     */
    public boolean initialize() {
        // For API level 18 and above, get a reference to BluetoothAdapter through
        // BluetoothManager.
        if (mBluetoothManager == null) {
            mBluetoothManager = (BluetoothManager) telemService.getSystemService(Context.BLUETOOTH_SERVICE);
            if (mBluetoothManager == null) {
            	if (ERROR) Log.e(TAG, "Unable to initialize BluetoothManager.");
                return false;
            }
        }

        mBluetoothAdapter = mBluetoothManager.getAdapter();
        if (mBluetoothAdapter == null) {
            if (ERROR) Log.e(TAG, "Unable to obtain a BluetoothAdapter.");
            return false;
        }

        return true;
    }

    /**
     * Connects to the GATT server hosted on the Bluetooth LE device.
     *
     * @param address The device address of the destination device.
     *
     * @return Return true if the connection is initiated successfully. The connection result
     *         is reported asynchronously through the
     *         {@code BluetoothGattCallback#onConnectionStateChange(android.bluetooth.BluetoothGatt, int, int)}
     *         callback.
     */
    public boolean btleConnect(final String address) {
        if (mBluetoothAdapter == null || address == null) {
            if (WARN) Log.w(TAG, "BluetoothAdapter not initialized or unspecified address.");
            return false;
        }

        // Previously connected device.  Try to reconnect.
        if (mBluetoothDeviceAddress != null && address.equals(mBluetoothDeviceAddress)
                && mBluetoothGatt != null) {
            if (DEBUG) Log.d(TAG, "Trying to use an existing mBluetoothGatt for connection.");
            if (mBluetoothGatt.connect()) {
                mConnectionState = STATE_CONNECTING;
                return true;
            } else {
                final BluetoothDevice device = mBluetoothAdapter.getRemoteDevice(address);
                mBluetoothGatt = device.connectGatt(telemService, false, mGattCallback);
                mBluetoothDeviceAddress = address;
                return false;
            }
        }

        final BluetoothDevice device = mBluetoothAdapter.getRemoteDevice(address);
        if (device == null) {
            if (WARN) Log.w(TAG, "Device not found.  Unable to connect.");
            return false;
        }
        // We want to directly connect to the device, so we are setting the autoConnect
        // parameter to false.
        mBluetoothGatt = device.connectGatt(telemService, false, mGattCallback);
        if (DEBUG) Log.d(TAG, "Trying to create a new connection.");
        mBluetoothDeviceAddress = address;
        mConnectionState = STATE_CONNECTING;
        return true;
    }

    /**
     * Disconnects an existing connection or cancel a pending connection. The disconnection result
     * is reported asynchronously through the
     * {@code BluetoothGattCallback#onConnectionStateChange(android.bluetooth.BluetoothGatt, int, int)}
     * callback.
     */
    public void btleDisconnect() {
        if (mBluetoothAdapter == null || mBluetoothGatt == null) {
            if (WARN) Log.w(TAG, "BluetoothAdapter not initialized");
            return;
        }
        mBluetoothGatt.disconnect();
    }

    /**
     * After using a given BLE device, the app must call this method to ensure resources are
     * released properly.
     */
    public void close() {
        if (mBluetoothGatt == null) {
            return;
        }
        mBluetoothGatt.close();
        mBluetoothGatt = null;
    }

    /**
     * Request a read on a given {@code BluetoothGattCharacteristic}. The read result is reported
     * asynchronously through the {@code BluetoothGattCallback#onCharacteristicRead(android.bluetooth.BluetoothGatt, android.bluetooth.BluetoothGattCharacteristic, int)}
     * callback.
     *
     * @param characteristic The characteristic to read from.
     */
    public void readCharacteristic(BluetoothGattCharacteristic characteristic) {
        if (mBluetoothAdapter == null || mBluetoothGatt == null) {
            if (WARN) Log.w(TAG, "BluetoothAdapter not initialized");
            return;
        }
        mBluetoothGatt.readCharacteristic(characteristic);
    }

    /**
     * Write to a given char
     * @param characteristic The characteristic to write to
     */
	public void writeCharacteristic(BluetoothGattCharacteristic characteristic) {
		if (mBluetoothAdapter == null || mBluetoothGatt == null) {
			if (WARN) Log.w(TAG, "BluetoothAdapter not initialized");
			return;
		}

		mBluetoothGatt.writeCharacteristic(characteristic);
	}   
    
    /**
     * Enables or disables notification on a give characteristic.
     *
     * @param characteristic Characteristic to act on.
     * @param enabled If true, enable notification.  False otherwise.
     */
    public void setCharacteristicNotification(BluetoothGattCharacteristic characteristic,
                                              boolean enabled) {
        if (mBluetoothAdapter == null || mBluetoothGatt == null) {
        	if (WARN) Log.w(TAG, "BluetoothAdapter not initialized");
            return;
        }
        mBluetoothGatt.setCharacteristicNotification(characteristic, enabled);

        // This is specific to Heart Rate Measurement.
        if (UUID_HM_RX_TX.equals(characteristic.getUuid())) {
            BluetoothGattDescriptor descriptor = characteristic.getDescriptor(
                    UUID.fromString(SampleGattAttributes.CLIENT_CHARACTERISTIC_CONFIG));
            descriptor.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
            mBluetoothGatt.writeDescriptor(descriptor);
        }
    }

    /**
     * Retrieves a list of supported GATT services on the connected device. This should be
     * invoked only after {@code BluetoothGatt#discoverServices()} completes successfully.
     *
     * @return A {@code List} of supported services.
     */
    public List<BluetoothGattService> getSupportedGattServices() {
        if (mBluetoothGatt == null) return null;

        return mBluetoothGatt.getServices();
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

				if (DEBUG) Log.d(TAG, "Sending " + n + " bytes starting at " + idx + " out of " + LEN);

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
			synchronized(data) {
				data.put(b);
				data.notify();
			}
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
