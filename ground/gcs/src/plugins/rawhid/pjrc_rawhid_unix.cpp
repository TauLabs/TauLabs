/**
 ******************************************************************************
 * @file       pjrc_rawhid_unix.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup RawHIDPlugin Raw HID Plugin
 * @{
 * @brief Impliments a HID USB connection to the flight hardware as a QIODevice
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

#include "pjrc_rawhid.h"

#define printf qDebug

pjrc_rawhid::pjrc_rawhid()
{
    int result;
    result = libusb_init(&m_pLibraryContext);
    if (result != 0) {
        fprintf(stderr, "pjrc_rawhid_unix: Unable to initialize libusb-1.0 %d\n", result);
        m_pLibraryContext = NULL;
        return;
    }

    libusb_set_debug(m_pLibraryContext, DEBUG_LEVEL);
}

pjrc_rawhid::~pjrc_rawhid()
{
    for (std::size_t iDevice = 0; iDevice < m_DeviceHandles.size(); ++iDevice) {
        if (m_DeviceHandles[iDevice] != NULL)
            close(iDevice);
    }

    if (m_pLibraryContext != NULL)
        libusb_exit(m_pLibraryContext);
}

//  open - open 1 or more devices
//
//    Inputs:
//    max = maximum number of devices to open
//    vid = Vendor ID, or -1 if any
//    pid = Product ID, or -1 if any
//    usage_page = unused, argument kept for api consistency
//    usage = unused, argument kept for api consistency
//    Output:
//    actual number of devices opened
//
int pjrc_rawhid::open(int max, int vid, int pid, int /* usage_page */, int /* usage */)
{
    //check if there are still open devices and close them
    if (!m_DeviceHandles.empty()) {
        for (std::size_t iDevice = 0; iDevice < m_DeviceHandles.size(); ++iDevice) {
            if (m_DeviceHandles[iDevice] != NULL)
                close(iDevice);
        }
        m_DeviceHandles.clear();
        m_DeviceInterfaces.clear();
    }

    int retval;

    //enumerate all devices
    libusb_device** list = NULL;
    ssize_t num_devices;
    num_devices = libusb_get_device_list(m_pLibraryContext, &list);

    for (ssize_t iDevice = 0;
        iDevice < num_devices && m_DeviceHandles.size() < (std::size_t)max;
        ++iDevice)
    {
        libusb_device_descriptor device_desc;
        retval = libusb_get_device_descriptor(list[iDevice], &device_desc);
        if (retval < 0) {
            fprintf(stderr, "pjrc_rawhid_unix: Failed to get device descriptor (%d)\n", retval);
            continue;
        }

        //check if we are interested in this device
        if (device_desc.idVendor != vid || device_desc.idProduct != pid)
            continue;

        //search for proper interface
        libusb_config_descriptor* config_desc;
        retval = libusb_get_config_descriptor(list[iDevice], 0, &config_desc);
        if (retval != 0) {
            fprintf(stderr, "pjrc_rawhid_unix: libusb_get_config_descriptor error (%d)\n", retval);
            continue;
        }

        int device_interface = -1;
        const libusb_interface* interface;
        const libusb_interface_descriptor* interface_desc;
        for (ssize_t iInterface = 0;
            iInterface < config_desc->bNumInterfaces && device_interface == -1;
            ++iInterface)
        {
            interface = &config_desc->interface[iInterface];

            for (ssize_t iAltSetting = 0;
                    iAltSetting < interface->num_altsetting && device_interface == -1;
                    ++iAltSetting)
            {
                interface_desc = &interface->altsetting[iAltSetting];

                if (interface_desc->bInterfaceClass == LIBUSB_CLASS_HID &&
                        interface_desc->bInterfaceSubClass == 0 &&
                        interface_desc->bInterfaceProtocol == 0)
                {
                    device_interface = iInterface;
                }
            }
        }

        libusb_free_config_descriptor(config_desc);

        //check if we found an interface, if not, skip device
        if (device_interface == -1)
            continue;

        libusb_device_handle* device_handle;
        retval = libusb_open(list[iDevice], &device_handle);
        //skip if device cant be opened
        if (retval != 0) {
            fprintf(stderr, "pjrc_rawhid_unix: Failed to open device (%d)\n", retval);
            continue;
        }

        //detach kernel
        retval = libusb_detach_kernel_driver(device_handle, device_interface);
        if (retval != 0 && retval != LIBUSB_ERROR_NOT_FOUND) {
            fprintf(stderr, "pjrc_rawhid_unix: Unable to detach kernel driver (%d)\n", retval);
            libusb_close(device_handle);
            continue;
        }

        //claim interface
        retval = libusb_claim_interface(device_handle, device_interface);
        if (retval != 0) {
            fprintf(stderr, "pjrc_rawhid_unix: libusb_claim_interface error (%d)\n", retval);
            libusb_close(device_handle);
            continue;
        }

        m_DeviceHandles.push_back(device_handle);
        m_DeviceInterfaces.push_back(device_interface);
    }

    libusb_free_device_list(list, 1);

    return m_DeviceHandles.size();
}

//  recveive - receive a packet
//    Inputs:
//    num = device to receive from (zero based)
//    buf = buffer to receive packet
//    len = buffer's size
//    timeout = time to wait, in milliseconds
//    Output:
//    number of bytes received, or -1 on error
//
int pjrc_rawhid::receive(int num, void *buf, int len, int timeout)
{
    if ((std::size_t)num >= m_DeviceHandles.size()) {
        fprintf(stderr, "pjrc_rawhid_unix: Invalid device number used (%d)\n", num);
        return -1;
    }
    if (m_DeviceHandles[num] == NULL) {
        fprintf(stderr, "pjrc_rawhid_unix: Tried to use a device which is not open (%d)\n", num);
        return -1;
    }

    int bytes_transferred;
    int retval;

    // Read data from the device.
    retval = libusb_interrupt_transfer(
        m_DeviceHandles[num],
        INTERRUPT_IN_ENDPOINT,
        (unsigned char*)buf,
        len > MAX_INTERRUPT_IN_TRANSFER_SIZE ? MAX_INTERRUPT_IN_TRANSFER_SIZE : len,
        &bytes_transferred,
        timeout);

    if (retval >= 0) {
        if (bytes_transferred > 0) {
            return bytes_transferred;
        }
        else {
            fprintf(stderr, "pjrc_rawhid_unix: No data received in interrupt transfer (%d)\n", retval);
            return -1;
        }
    }
    else {
        if (retval == LIBUSB_ERROR_TIMEOUT)
            return 0;
        fprintf(stderr, "pjrc_rawhid_unix: Error receiving data via interrupt transfer (%d)\n", retval);
        return -1;
    }
}

//  send - send a packet
//    Inputs:
//    num = device to transmit to (zero based)
//    buf = buffer containing packet to send
//    len = number of bytes to transmit
//    timeout = time to wait, in milliseconds
//    Output:
//    number of bytes sent, or -1 on error
//
int pjrc_rawhid::send(int num, void *buf, int len, int timeout)
{
    if ((std::size_t)num >= m_DeviceHandles.size()) {
        fprintf(stderr, "pjrc_rawhid_unix: Invalid device number used (%d)\n", num);
        return -1;
    }
    if (m_DeviceHandles[num] == NULL) {
        fprintf(stderr, "pjrc_rawhid_unix: Tried to use a device which is not open (%d)\n", num);
        return -1;
    }

    int bytes_transferred;
    int retval;

    // Write data to the device.
    retval = libusb_interrupt_transfer(
        m_DeviceHandles[num],
        INTERRUPT_OUT_ENDPOINT,
        (unsigned char*)buf,
        len > MAX_INTERRUPT_OUT_TRANSFER_SIZE ? MAX_INTERRUPT_OUT_TRANSFER_SIZE : len,
        &bytes_transferred,
        timeout);

    if (retval >= 0) {
        return bytes_transferred;
    }
    else {
        fprintf(stderr, "pjrc_rawhid_unix: Error sending data via interrupt transfer (%d)\n", retval);
        return -1;
    }
}

//  getserial - get the serialnumber of the device
//
//    Inputs:
//    num = device to get serial from
//    Output
//    QString conatining device serial or empty QString
//
QString pjrc_rawhid::getserial(int num)
{
    if ((std::size_t)num >= m_DeviceHandles.size()) {
        fprintf(stderr, "pjrc_rawhid_unix: Invalid device number used (%d)\n", num);
        return "";
    }
    if (m_DeviceHandles[num] == NULL) {
        fprintf(stderr, "pjrc_rawhid_unix: Tried to use a device which is not open (%d)\n", num);
        return "";
    }

    struct libusb_device_descriptor desc;
    struct libusb_device * dev = libusb_get_device(m_DeviceHandles[num]);
    int retval;

    retval = libusb_get_device_descriptor(dev, &desc);
    if (retval < 0) {
        fprintf(stderr, "pjrc_rawhid_unix: Failed to get device descriptor (%d)\n", retval);
        return "";
    }

    unsigned char buf[128];
    retval = libusb_get_string_descriptor_ascii(m_DeviceHandles[num], desc.iSerialNumber, buf, sizeof(buf));
    if (retval < 0) {
        fprintf(stderr, "pjrc_rawhid_unix: Coudn't get serial string (%d)\n", retval);
        return "";
    }

    return QString().fromAscii((char*)buf,-1);
}

//  close - close a device
//
//    Inputs:
//    num = device to close (zero based)
//    Output
//    (nothing)
//
void pjrc_rawhid::close(int num)
{
    if ((std::size_t)num >= m_DeviceHandles.size()) {
        fprintf(stderr, "pjrc_rawhid_unix: Invalid device number used (%d)\n", num);
        return;
    }
    if (m_DeviceHandles[num] == NULL) {
        fprintf(stderr, "pjrc_rawhid_unix: Tried to use a device which is not open (%d)\n", num);
        return;
    }

    int retval;
    retval = libusb_release_interface(m_DeviceHandles[num], m_DeviceInterfaces[num]);
    if (retval != 0) {
        fprintf(stderr, "pjrc_rawhid_unix: Unable to release interface (%d)\n", retval != 0);
    }

    libusb_close(m_DeviceHandles[num]);

    m_DeviceHandles[num] = NULL;
}

