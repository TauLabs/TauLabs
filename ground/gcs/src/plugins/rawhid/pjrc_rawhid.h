/**
 ******************************************************************************
 *
 * @file       pjrc_rawhid.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://github.com/TauLabs, Copyright (C) 2013.
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

#ifndef PJRC_RAWHID_H
#define PJRC_RAWHID_H

#include <stdio.h>
#include <stdlib.h>
#include <qglobal.h>
#include <vector>
#include <QDebug>
#include <QMutex>
#include <QString>
#include "rawhid_global.h"

#if defined( Q_OS_MAC)

#include <IOKit/IOKitLib.h>
#include <IOKit/hid/IOHIDLib.h>
#include <CoreFoundation/CFString.h>

#elif defined(Q_OS_UNIX)
#include <libusb-1.0/libusb.h>
#include <QDebug>
#include <QString>
#elif defined(Q_OS_WIN32)
#include <windows.h>
#include <setupapi.h>
#include <hidsdi.h>

#ifdef __cplusplus
extern "C" {
#endif
// Some functions no longer included in hidsdi.h

//HIDAPI BOOL NTAPI HidD_GetAttributes (HANDLE, PHIDD_ATTRIBUTES);
HIDAPI VOID NTAPI HidD_GetHidGuid (LPGUID);
HIDAPI BOOL NTAPI HidD_GetPreparsedData(HANDLE, PHIDP_PREPARSED_DATA  *);
HIDAPI BOOL NTAPI HidD_FreePreparsedData(PHIDP_PREPARSED_DATA);
HIDAPI BOOL NTAPI HidD_FlushQueue (HANDLE);
HIDAPI BOOL NTAPI HidD_GetConfiguration (HANDLE, PHIDD_CONFIGURATION, ULONG);
HIDAPI BOOL NTAPI HidD_SetConfiguration (HANDLE, PHIDD_CONFIGURATION, ULONG);
//HIDAPI BOOL NTAPI HidD_GetFeature (HANDLE, PVOID, ULONG);
//HIDAPI BOOL NTAPI HidD_SetFeature (HANDLE, PVOID, ULONG);
HIDAPI BOOL NTAPI HidD_GetNumInputBuffers (HANDLE, PULONG);
HIDAPI BOOL NTAPI HidD_SetNumInputBuffers (HANDLE HidDeviceObject, ULONG);
HIDAPI BOOL NTAPI HidD_GetPhysicalDescriptor (HANDLE, PVOID, ULONG);
HIDAPI BOOL NTAPI HidD_GetManufacturerString (HANDLE, PVOID, ULONG);
HIDAPI BOOL NTAPI HidD_GetProductString ( HANDLE, PVOID, ULONG);
HIDAPI BOOL NTAPI HidD_GetIndexedString ( HANDLE, ULONG, PVOID, ULONG);
HIDAPI BOOL NTAPI HidD_GetSerialNumberString (HANDLE, PVOID, ULONG);

#ifdef __cplusplus
}
#endif
#endif

// ************

#if defined( Q_OS_MAC)

// todo:

#elif defined(Q_OS_UNIX)


#elif defined(Q_OS_WIN32)

typedef struct hid_struct hid_t;

struct hid_struct
{
    HANDLE handle;
    int open;
    struct hid_struct *prev;
    struct hid_struct *next;
};

#endif




class RAWHID_EXPORT pjrc_rawhid: public QObject
{
    Q_OBJECT

public:
    pjrc_rawhid();
    ~pjrc_rawhid();
    int open(int max, int vid, int pid, int usage_page, int usage);
    int receive(int, void *buf, int len, int timeout);
    void close(int num);
    int send(int num, void *buf, int len, int timeout);
    QString getserial(int num);
signals:
     void deviceUnplugged(int);

private:
#if defined( Q_OS_MAC)

     // Static callbacks called by the HID system with handles to the PJRC object
     static void attach_callback(void *, IOReturn, void *, IOHIDDeviceRef);
     static void dettach_callback(void *, IOReturn, void *hid_mgr, IOHIDDeviceRef dev);
     static void input_callback(void *, IOReturn, void *, IOHIDReportType, quint32, quint8 *, CFIndex);
     static void timeout_callback(CFRunLoopTimerRef, void *);

     // Non static methods to call into
     void attach(IOHIDDeviceRef dev);
     void dettach(IOHIDDeviceRef dev);
     void input(quint8 *, CFIndex);

     // Platform specific handles for the USB device
     IOHIDManagerRef hid_manager;
     IOHIDDeviceRef dev;
     CFRunLoopRef the_correct_runloop;
     CFRunLoopRef received_runloop;

     static const int BUFFER_SIZE = 64;
     quint8 buffer[BUFFER_SIZE];
     int attach_count;
     int buffer_count;
     bool device_open;
     bool unplugged;

     QMutex *m_writeMutex;
     QMutex *m_readMutex;
#elif defined(Q_OS_UNIX)
     // Assumes interrupt endpoint 2 IN and OUT:
     static const int INTERRUPT_IN_ENDPOINT = 0x81;
     static const int INTERRUPT_OUT_ENDPOINT = 0x01;

     // With firmware support, transfers can be > the endpoint's max packet size.
     static const int MAX_INTERRUPT_IN_TRANSFER_SIZE = 64;
     static const int MAX_INTERRUPT_OUT_TRANSFER_SIZE = 64;

     //libusb debug level
     //Level 0: no messages ever printed by the library (default)
     //Level 1: error messages are printed to stderr
     //Level 2: warning and error messages are printed to stderr
     //Level 3: informational messages are printed to stdout, warning and error messages are printed to stderr
     static const int DEBUG_LEVEL = 1;

     libusb_context* m_pLibraryContext;
     std::vector<libusb_device_handle*> m_DeviceHandles;
     std::vector<ssize_t> m_DeviceInterfaces;


     int hid_parse_item(quint32 *val, quint8 **data, const quint8 *end);

#elif defined(Q_OS_WIN32)

    hid_t *first_hid;
    hid_t *last_hid;
    HANDLE rx_event;
    HANDLE tx_event;
    CRITICAL_SECTION rx_mutex;
    CRITICAL_SECTION tx_mutex;

    void add_hid(hid_t *h);
    hid_t * get_hid(int num);
    void free_all_hid(void);
    void hid_close(hid_t *hid);
    void print_win32_err(DWORD err);

#endif
};

#endif
