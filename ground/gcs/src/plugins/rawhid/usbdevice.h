#ifndef USBDEVICE_H
#define USBDEVICE_H

#include <coreplugin/idevice.h>

class USBDevice : public Core::IDevice
{
public:
    USBDevice();

    // Our USB-specific device info:
    int getVendorID() { return vendorID; }
    int getProductID() { return productID; }
    void setVendorID(int vid) { vendorID = vid; }
    void setProdictID(int pid) { productID = pid; }

//private:
    int vendorID;
    int productID;

};

#endif // USBDEVICE_H
