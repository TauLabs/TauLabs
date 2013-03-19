/* Copyright 2011 Google Inc.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
 * USA.
 *
 * Project home page: http://code.google.com/p/usb-serial-for-android/
 */

package com.hoho.android.usbserial.driver;

import java.util.Map;

import android.app.PendingIntent;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbManager;
import android.os.Handler;
import android.util.Log;

/**
 * Helper class to assist in detecting and building {@link UsbSerialDriver}
 * instances from available hardware.
 *
 * @author mike wakerly (opensource@hoho.com)
 */
public enum UsbSerialProber {

    // TODO(mikey): Too much boilerplate.

    /**
     * Prober for {@link FtdiSerialDriver}.
     *
     * @see FtdiSerialDriver
     */
    FTDI_SERIAL {
        @Override
        public boolean getDevice(final UsbManager manager, final Service service, final UsbDevice usbDevice) {
            if (!testIfSupported(usbDevice, FtdiSerialDriver.getSupportedDevices())) {
                return false;
            }

    		permissionIntent = PendingIntent.getBroadcast(service, 0, new Intent(ACTION_USB_PERMISSION), 0);
    		permissionFilter = new IntentFilter(ACTION_USB_PERMISSION);
    		service.registerReceiver(usbPermissionReceiver, permissionFilter);

    		UsbSerialProber.manager = manager;
    		manager.requestPermission(usbDevice, permissionIntent);

    		return true;
        }
    },

    CDC_ACM_SERIAL {
        @Override
        public boolean getDevice(UsbManager manager, final Service service, UsbDevice usbDevice) {
            if (!testIfSupported(usbDevice, CdcAcmSerialDriver.getSupportedDevices())) {
               return false;
            }
            final UsbDeviceConnection connection = manager.openDevice(usbDevice);
            if (connection == null) {
                return false;
            }
            return true; //new CdcAcmSerialDriver(usbDevice, connection);
        }
    },

    SILAB_SERIAL {
        @Override
        public boolean getDevice(final UsbManager manager, final Service service, final UsbDevice usbDevice) {
            if (!testIfSupported(usbDevice, Cp2102SerialDriver.getSupportedDevices())) {
                return false;
            }
            final UsbDeviceConnection connection = manager.openDevice(usbDevice);
            if (connection == null) {
                return false;
            }
            return true; //new Cp2102SerialDriver(usbDevice, connection);
        }
    };

    /**
     * Builds a new {@link UsbSerialDriver} instance from the raw device, or
     * returns <code>null</code> if it could not be built (for example, if the
     * probe failed).
     *
     * @param manager the {@link UsbManager} to use
     * @param usbDevice the raw {@link UsbDevice} to use
     * @return the first available {@link UsbSerialDriver}, or {@code null} if
     *         no devices could be acquired
     */
    public abstract boolean getDevice(final UsbManager manager, final Service service, final UsbDevice usbDevice);

    private static final String ACTION_USB_PERMISSION = "com.access.device.USB_PERMISSION";
	private static PendingIntent permissionIntent;
	private static IntentFilter permissionFilter;
	private final static String TAG = "UsbSerialProber";
	private final static boolean DEBUG = true;
	private static UsbDevice currentDevice;
	protected static UsbManager manager;
	public static FtdiSerialDriver driver = null;

	/*
	 * Receives a requested broadcast from the operating system.
	 * In this case the following actions are handled:
	 *   USB_PERMISSION
	 *   UsbManager.ACTION_USB_DEVICE_DETACHED
	 *   UsbManager.ACTION_USB_DEVICE_ATTACHED
	 */
	private final static BroadcastReceiver usbPermissionReceiver = new BroadcastReceiver()
	{
		@Override
		public void onReceive(Context context, Intent intent)
		{
			Log.d(TAG,"Broadcast receiver caught intent: " + intent);
			String action = intent.getAction();
			// Validate the action against the actions registered
			if (ACTION_USB_PERMISSION.equals(action))
			{
				// A permission response has been received, validate if the user has
				// GRANTED, or DENIED permission
				synchronized (this)
				{
					UsbDevice deviceConnected = (UsbDevice)intent.getParcelableExtra(UsbManager.EXTRA_DEVICE);

					if (DEBUG) Log.d(TAG, "Device Permission requested" + deviceConnected);
					if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false))
					{
						// Permission has been granted, so connect to the device
						// If this fails, then keep looking
						if (deviceConnected != null)
						{
							// call method to setup device communication
							currentDevice = deviceConnected;
							if (DEBUG) Log.d(TAG, "Device Permission Acquired" + currentDevice);
							Log.d(TAG, "Connect to device here");

				            final UsbDeviceConnection connection = manager.openDevice(currentDevice);
				            driver = new FtdiSerialDriver(currentDevice, connection);

				            handler.post(connectedRunnable);
						}
					}
					else
					{
						// Permission has not been granted, so keep looking for another
						// device to be attached....
						if (DEBUG) Log.d(TAG, "Device Permission Denied" + deviceConnected);
						currentDevice = null;
					}
				}
			}
		}
	};

    /**
     * Acquires and returns the first available serial device among all
     * available {@link UsbDevice}s, or returns {@code null} if no device could
     * be acquired.
     *
     * @param usbManager the {@link UsbManager} to use
     * @return the first available {@link UsbSerialDriver}, or {@code null} if
     *         no devices could be acquired
     */
    public static boolean acquire(final UsbManager usbManager, final Service service) {
    	driver = null;
        for (final UsbDevice usbDevice : usbManager.getDeviceList().values()) {
            final boolean probedDevice = acquire(usbManager, service, usbDevice);
            if (probedDevice != false) {
                return probedDevice;
            }
        }
        return false;
    }

    /**
     * Builds and returns a new {@link UsbSerialDriver} from the given
     * {@link UsbDevice}, or returns {@code null} if no drivers supported this
     * device.
     *
     * @param usbManager the {@link UsbManager} to use
     * @param usbDevice the {@link UsbDevice} to use
     * @return a new {@link UsbSerialDriver}, or {@code null} if no devices
     *         could be acquired
     */
    public static boolean acquire(final UsbManager usbManager, final Service service, final UsbDevice usbDevice) {
        for (final UsbSerialProber prober : values()) {
            final boolean probedDevice = prober.getDevice(usbManager, service, usbDevice);
            if (probedDevice != false) {
                return probedDevice;
            }
        }
        return false;
    }

    /**
     * Returns {@code true} if the given device is found in the vendor/product map.
     *
     * @param usbDevice the device to test
     * @param supportedDevices map of vendor ids to product id(s)
     * @return {@code true} if supported
     */
    private static boolean testIfSupported(final UsbDevice usbDevice,
            final Map<Integer, int[]> supportedDevices) {
        final int[] supportedProducts = supportedDevices.get(
                Integer.valueOf(usbDevice.getVendorId()));
        if (supportedProducts == null) {
            return false;
        }

        final int productId = usbDevice.getProductId();
        for (int supportedProductId : supportedProducts) {
            if (productId == supportedProductId) {
                return true;
            }
        }
        return false;
    }

    private static Runnable connectedRunnable;
    private static Handler handler;
    public static void setConnectedRunnable(Runnable r, Handler h)
    {
    	connectedRunnable = r;
    	handler = h;
    }
}
