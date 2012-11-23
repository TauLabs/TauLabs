/* @file pjrc_rawhid_mac.cpp
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup RawHIDPlugin Raw HID Plugin
 * @{
 * @brief Impliments a HID USB connection to the flight hardware as a QIODevice
 *****************************************************************************/

/* Simple Raw HID functions for Linux - for use with Teensy RawHID example
 * http://www.pjrc.com/teensy/rawhid.html
 * Copyright (c) 2009 PJRC.COM, LLC
 *
 *  rawhid_open - open 1 or more devices
 *  rawhid_recv - receive a packet
 *  rawhid_send - send a packet
 *  rawhid_close - close a device
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above description, website URL and copyright notice and this permission
 * notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * Version 1.0: Initial Release
 */

/*
 * There are a lot of threading issues with OSX HID so we need to make
 * sure all the USB communications runs on a single thread.  Unfortunately
 * to make that thread work properly it needs to run a CFRunLoop which
 * precludes running a QEventLoop so we cannot receive Qt signals.
 *
 * In principle the QObject _should_ be able to send signals, so whenever
 * data arrives that signal can be emitted.
 *
 * To know when to send data we need to make a CFRunLoopSource which the
 * send method will notify.
 */

#include "pjrc_rawhid.h"

#include <unistd.h>
#include <QString>
#include <QThread>
#include <QTimer>
#include <QCoreApplication>

struct timeout_info {
    CFRunLoopRef loopRef;
    bool timed_out;
};

pjrc_rawhid::pjrc_rawhid() :
    device_open(false), hid_manager(NULL), recv_buffer_count(0), unplugged(false), close_requested(false)
{
}

pjrc_rawhid::~pjrc_rawhid()
{
    if (device_open) {
        close(0);
    }
}

/**
 * @brief pjrc_rawhid::open Attempt to open the device matching the class
 * settings and register against this thread.  Must be run from within
 * the thread.
 */
int pjrc_rawhid::open() {

    CFMutableDictionaryRef dict;
    CFNumberRef num;
    IOReturn ret;

    Q_ASSERT(hid_manager == NULL);
    Q_ASSERT(device_open == false);

    qDebug() << "Thread CFRunLoop: " << CFRunLoopGetCurrent();

    attach_count = 0;

    // Start the HID Manager
    hid_manager = IOHIDManagerCreate(kCFAllocatorDefault, kIOHIDOptionsTypeNone);
    if (hid_manager == NULL || CFGetTypeID(hid_manager) != IOHIDManagerGetTypeID()) {
        if (hid_manager) CFRelease(hid_manager);
        return 0;
    }

    if (vid > 0 || pid > 0 || usage_page > 0 || usage > 0) {
        // Tell the HID Manager what type of devices we want
        dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                         &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        if (!dict) return 0;
        if (vid > 0) {
            num = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &vid);
            CFDictionarySetValue(dict, CFSTR(kIOHIDVendorIDKey), num);
            CFRelease(num);
        }
        if (pid > 0) {
            num = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &pid);
            CFDictionarySetValue(dict, CFSTR(kIOHIDProductIDKey), num);
            CFRelease(num);
        }
        if (usage_page > 0) {
            num = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &usage_page);
            CFDictionarySetValue(dict, CFSTR(kIOHIDPrimaryUsagePageKey), num);
            CFRelease(num);
        }
        if (usage > 0) {
            num = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &usage);
            CFDictionarySetValue(dict, CFSTR(kIOHIDPrimaryUsageKey), num);
            CFRelease(num);
        }
        IOHIDManagerSetDeviceMatching(hid_manager, dict);
        CFRelease(dict);
    } else {
        IOHIDManagerSetDeviceMatching(hid_manager, NULL);
    }

    // Set the run loop reference before configuring the attach callback
    runloop = CFRunLoopGetCurrent();

    // set up a callbacks for device attach
    IOHIDManagerScheduleWithRunLoop(hid_manager, runloop, kCFRunLoopDefaultMode);
    IOHIDManagerRegisterDeviceMatchingCallback(hid_manager, pjrc_rawhid::attach_callback, this);
    ret = IOHIDManagerOpen(hid_manager, kIOHIDOptionsTypeNone);
    if (ret != kIOReturnSuccess) {
        IOHIDManagerUnscheduleFromRunLoop(hid_manager, runloop, kCFRunLoopDefaultMode);
        CFRelease(hid_manager);
        return 0;
    }

    // let it do the callback for all devices
    while (CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, true) == kCFRunLoopRunHandledSource) ;

    qDebug() << "Attached: " << attach_count;
    return attach_count;
}

/**
 * @brief pjrc_rawhid::run
 */
void pjrc_rawhid::run() {
    open();

    // Strategy:
    // 1. running the run loop will receive usb message and then notify anything
    //    calling receive
    // 2. check for any requests to send data and if they are there call set report
    while(attach_count && device_open) {
        // This will get any incoming reports
        while (CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, true) == kCFRunLoopRunHandledSource) ;

        // If pending send requests then send them
        if (send_buffer_count != 0 && !unplugged) {
            IOHIDDeviceSetReport(dev, kIOHIDReportTypeOutput, 2, send_buffer, send_buffer_count);
            emit sendComplete(send_buffer_count);
            send_buffer_count = 0;
        }

        if (close_requested)
            close();
    }

    qDebug() << "USB thread terminated";
}

/**
  * @brief open - open 1 or more devices
  * @param[in] max maximum number of devices to open
  * @param[in] vid Vendor ID, or -1 if any
  * @param[in] pid Product ID, or -1 if any
  * @param[in] usage_page top level usage page, or -1 if any
  * @param[in] usage top level usage number, or -1 if any
  * @returns actual number of devices opened
  */
int pjrc_rawhid::open(int max, int vid, int pid, int usage_page, int usage)
{
    Q_UNUSED(max);
    this->vid = vid;
    this->pid = pid;
    this->usage_page = usage_page;
    this->usage = usage;
    attach_count = 0;

    qDebug() << "Open CFRunLoop: " << CFRunLoopGetCurrent();

    start();

    while(attach_count == 0);

    qDebug() << "Finished opening";

    return 1;
}

/**
 * @brief receive - receive a packet
 * @param[in] num device to receive from (unused now)
 * @param[in] buf buffer to receive packet
 * @param[in] len buffer's size
 * @param[in] timeout = time to wait, in milliseconds
 * @returns number of bytes received, or -1 on error
 */
int pjrc_rawhid::receive(int, void *buf, int len, int timeout)
{
    if (!device_open)
        return -1;

    // If no data either wait for it or the timeout
    if (recv_buffer_count == 0) {
        QEventLoop el;
        QTimer::singleShot(timeout, &el, SLOT(quit()));
        connect(this,SIGNAL(receiveComplete(int)), &el, SLOT(quit()));
        el.exec();
    }

    if (recv_buffer_count != 0) {
        if (len > recv_buffer_count) len = recv_buffer_count;
        memcpy(buf, recv_buffer, len);
        recv_buffer_count = 0;
        return len;
    }

    return -1;
}

/**
 * @brief send - send a packet
 * @param[in] num device to transmit to (zero based)
 * @param[in] buf buffer containing packet to send
 * @param[in] len number of bytes to transmit
 * @param[in] timeout = time to wait, in milliseconds
 * @returns number of bytes sent, or -1 on error
 *
 * @Note: This method must be run from a thread running an event
 * a QT Event loop to receive the timeout signals
 */
int pjrc_rawhid::send(int, void *buf, int len, int timeout)
{
    if(!device_open || unplugged) {
        return -1;
    }

    if (send_buffer_count != 0) {
        QEventLoop el;
        QTimer::singleShot(timeout, &el, SLOT(quit()));
        connect(this,SIGNAL(sendComplete(int)), &el, SLOT(quit()));
        el.exec();
    }

    if (send_buffer_count != 0)
        return -1;

    memcpy(&send_buffer[0], buf, len);
    send_buffer_count = len;

    return len;
}

//! Get the serial number for a HID device
QString pjrc_rawhid::getserial(int num) {
    Q_UNUSED(num);

    if (!device_open || unplugged)
        return "";

    CFTypeRef serialnum = IOHIDDeviceGetProperty(dev, CFSTR(kIOHIDSerialNumberKey));
    if(serialnum && CFGetTypeID(serialnum) == CFStringGetTypeID())
    {
        //Note: I'm not sure it will always succeed if encoded as MacRoman but that
        //is a superset of UTF8 so I think this is fine
        CFStringRef str = (CFStringRef)serialnum;
        const char * buf = CFStringGetCStringPtr(str, kCFStringEncodingMacRoman);
        return QString(buf);
    }

    return QString("Error");
}

//! Close the HID device
void pjrc_rawhid::close(int)
{
    qDebug() << "Close requested";
    close_requested = true;
    while(device_open);
}

void pjrc_rawhid::close() {

    qDebug() << "Calling close";
    if (device_open) {
        CFRunLoopStop(runloop);

        if (!unplugged) {
            IOHIDDeviceUnscheduleFromRunLoop(dev, runloop, kCFRunLoopDefaultMode);
            IOHIDDeviceRegisterInputReportCallback(dev, recv_buffer, sizeof(recv_buffer), NULL, NULL);
            IOHIDDeviceClose(dev, kIOHIDOptionsTypeNone);
        }

        IOHIDManagerRegisterDeviceRemovalCallback(hid_manager, NULL, NULL);
        IOHIDManagerClose(hid_manager, 0);

        dev = NULL;
        hid_manager = NULL;

        device_open = false;
    }
}

/**
 * @brief input Called to add input data to the buffer
 * @param[in] id Report id
 * @param[in] data The data buffer
 * @param[in] len The report length
 */
void pjrc_rawhid::input(uint8_t *data, CFIndex len)
{
    if (!device_open)
        return;

    if (len > BUFFER_SIZE) len = BUFFER_SIZE;
    // Note: packet preprocessing done in OS independent code
    memcpy(recv_buffer, &data[0], len);
    recv_buffer_count = len;

    emit receiveComplete(len);
}

//! Callback for the HID driver on an input report
void pjrc_rawhid::input_callback(void *c, IOReturn ret, void *sender, IOHIDReportType type, uint32_t id, uint8_t *data, CFIndex len)
{
    Q_UNUSED(sender);
    Q_UNUSED(type);
    Q_UNUSED(id);

    if (ret != kIOReturnSuccess || len < 1) return;

    pjrc_rawhid *context = (pjrc_rawhid *) c;
    context->input(data, len);
}

//! Timeout used for the
void pjrc_rawhid::timeout_callback(CFRunLoopTimerRef, void *i)
{
    struct timeout_info *info = (struct timeout_info *) i;
    info->timed_out = true;
    CFRunLoopStop(info->loopRef);
}

//! Called on a dettach event
void pjrc_rawhid::dettach(IOHIDDeviceRef d)
{
    //close();
    unplugged = true;
    device_open = false;
    if (d == dev)
        emit deviceUnplugged(0);
}

//! Called from the USB system and forwarded to the instance (context)
void pjrc_rawhid::dettach_callback(void *context, IOReturn, void *, IOHIDDeviceRef dev)
{
    qDebug() << "Dettach callback";
    pjrc_rawhid *p = (pjrc_rawhid*) context;
    p->dettach(dev);
}

/**
 * @brief Called by the USB system
 * @param dev The device that was attached
 */
void pjrc_rawhid::attach(IOHIDDeviceRef d)
{
    // Store the device handle
    dev = d;

    if (IOHIDDeviceOpen(dev, kIOHIDOptionsTypeNone) != kIOReturnSuccess) return;
    // Disconnect the attach callback since we don't want to automatically reconnect
    IOHIDManagerRegisterDeviceMatchingCallback(hid_manager, NULL, NULL);
    IOHIDDeviceScheduleWithRunLoop(dev, runloop, kCFRunLoopDefaultMode);
    IOHIDDeviceRegisterInputReportCallback(dev, recv_buffer, sizeof(recv_buffer), pjrc_rawhid::input_callback, this);
    IOHIDManagerRegisterDeviceRemovalCallback(hid_manager, pjrc_rawhid::dettach_callback, this);

    attach_count++;
    device_open = true;
    unplugged = false;
}

//! Called from the USB system and forwarded to the instance (context)
void pjrc_rawhid::attach_callback(void *context, IOReturn r, void *hid_mgr, IOHIDDeviceRef dev)
{
    Q_UNUSED(hid_mgr);
    Q_UNUSED(r);

    pjrc_rawhid *p = (pjrc_rawhid*) context;
    p->attach(dev);
}

