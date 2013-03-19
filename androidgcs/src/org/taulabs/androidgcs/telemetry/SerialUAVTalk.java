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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;

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

	//! The stream that receives data from the HID device
	private TalkInputStream inTalkStream;
	//! The stream which sends data to the HID device
	private TalkOutputStream outTalkStream;
	//! The thread which reads from the device to @ref inTalkStream
	private Thread readThread;
	//! The thread which reads from @ref outTalkStream to the device
	private Thread writeThread;

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
				openDevice(UsbSerialProber.driver);
			}
		}, handler);

		mUsbManager = (UsbManager) telemService.getSystemService(Context.USB_SERVICE);
		return UsbSerialProber.acquire(mUsbManager, telemService);
	}

	public boolean openDevice(UsbSerialDriver driver) {
		mSerialDevice = driver;

		inTalkStream = new TalkInputStream();
		outTalkStream = new TalkOutputStream();
		inStream = inTalkStream;
		outStream = outTalkStream;

		if (readThread != null || writeThread != null) {
			Log.e(TAG, "Already running HID???");
		}

		readThread = new Thread(readThreadRunnable, "Serial Read");
		readThread.start();

		writeThread = new Thread(writeThreadRunnable, "Serial Write");
		writeThread.start();

		telemService.toastMessage("Serial Device Opened");

		attemptSucceeded();
		return true;
	}

	@Override
	public void disconnect() {
		super.disconnect();
	}

	final Runnable readThreadRunnable = new Runnable() {
		@Override
		public void run() {
			byte d[] = new byte[1000];
			while (!shutdown) {
				// Read data and push it on the data buffer
				int bytes = 0;
				try {
					bytes = mSerialDevice.read(d,100);
				} catch (IOException e) {
					if (shutdown) {
						if (DEBUG) Log.d(TAG, "Thread interrupted.  Shutting down");
					} else {
						if (ERROR) Log.e(TAG, "Got unexpected interrupting in HID write", new Exception());
					}
				}
				if (bytes > 0)
					inTalkStream.write(Arrays.copyOfRange(d,0,bytes-1));
			}
		}
	};

	final Runnable writeThreadRunnable = new Runnable() {
		@Override
		public void run() {
			byte d[] = new byte[1000];
			while(!shutdown) {
				// Write data from the output stream to the device
				int bytes;
				try {
					bytes = outTalkStream.read(d);
					if (DEBUG) Log.d(TAG, "Wrote " + bytes + " to serial device");
				} catch (InterruptedException e) {
					if (shutdown) {
						if (DEBUG) Log.d(TAG, "Thread interrupted.  Shutting down");
					} else {
						if (ERROR) Log.e(TAG, "Got unexpected interrupting in HID write", new Exception());
					}
					break;
				}
				if (bytes > 0)
					try {
						mSerialDevice.write(Arrays.copyOfRange(d,0,bytes-1),100);
					} catch (IOException e) {
						if (shutdown) {
							if (DEBUG) Log.d(TAG, "Thread interrupted.  Shutting down");
						} else {
							if (ERROR) Log.e(TAG, "Got unexpected interrupting in HID write", new Exception());
						}
					}
			}
		}
	};


	/*********** Helper classes for telemetry streams ************/

	class TalkOutputStream extends OutputStream {
		// Uses ByteFifo.get()
		// and  ByteFifo.put(byte [])
		ByteFifo data = new ByteFifo();

		//! Blocks until data is available and store in the byte array
		public int read(byte[] dst) throws InterruptedException {
			synchronized(data) {
				if (data.remaining() == 0)
					data.wait();

				int size = Math.min(data.remaining(), dst.length);
				data.get(dst,0,size);
				return size;
			}
		}

		@Override
		public void write(int oneByte) throws IOException {
			// Throw exception when try and read after shutdown
			if (shutdown)
				throw new IOException();

			synchronized(data) {
				data.put((byte) oneByte);
				data.notify();
			}
		}

		@Override
		public void write(byte[] b) throws IOException {
			if (shutdown)
				throw new IOException();

			synchronized(data) {
				data.put(b);
				data.notify();
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
