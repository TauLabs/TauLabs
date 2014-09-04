#include "usbsignalfilter.h"
#include <QDebug>
void USBSignalFilter::m_deviceDiscovered(USBPortInfo port)
{
    availableDevices = USBMonitor::instance()->availableDevices();
    if((m_vid.contains(port.vendorID) || m_vid.isEmpty()) && (port.productID==m_pid || m_pid==-1) && ((port.bcdDevice>>8)==m_boardModel || m_boardModel==-1) &&
            ( (port.bcdDevice&0x00ff) ==m_runState || m_runState==-1))
    {
        qDebug()<<"USBSignalFilter emit device discovered";
        emit deviceDiscovered();
    }
}

void USBSignalFilter::m_deviceRemoved(USBPortInfo port)
{
    foreach (USBPortInfo knownPort, availableDevices) {
        if(!USBMonitor::instance()->availableDevices().contains(knownPort))
        {
            port = knownPort;
            if((m_vid.contains(port.vendorID) || m_vid.isEmpty()) && (port.productID==m_pid || m_pid==-1) && ((port.bcdDevice>>8)==m_boardModel || m_boardModel==-1) &&
                    ( (port.bcdDevice&0x00ff) ==m_runState || m_runState==-1))
            {
                qDebug()<<"USBSignalFilter emit device removed";
                emit deviceRemoved();
            }
        }
    }
    availableDevices = USBMonitor::instance()->availableDevices();
}

USBSignalFilter::USBSignalFilter(int vid, int pid, int boardModel, int runState):
    m_pid(pid),
    m_boardModel(boardModel),
    m_runState(runState)
{
    availableDevices = USBMonitor::instance()->availableDevices();
    m_vid.append(vid);
    connect(USBMonitor::instance(), SIGNAL(deviceDiscovered(USBPortInfo)), this, SLOT(m_deviceDiscovered(USBPortInfo)));
    connect(USBMonitor::instance(), SIGNAL(deviceRemoved(USBPortInfo)), this, SLOT(m_deviceRemoved(USBPortInfo)));
}

USBSignalFilter::USBSignalFilter(QList<int> vid, int pid, int boardModel, int runState):
    m_vid(vid),
    m_pid(pid),
    m_boardModel(boardModel),
    m_runState(runState)
{
    availableDevices = USBMonitor::instance()->availableDevices();
    connect(USBMonitor::instance(), SIGNAL(deviceDiscovered(USBPortInfo)), this, SLOT(m_deviceDiscovered(USBPortInfo)));
    connect(USBMonitor::instance(), SIGNAL(deviceRemoved(USBPortInfo)), this, SLOT(m_deviceRemoved(USBPortInfo)));
}
