#ifndef USBSIGNALFILTER_H
#define USBSIGNALFILTER_H
#include <QObject>
#include "usbmonitor.h"
#include <QList>
class RAWHID_EXPORT USBSignalFilter : public QObject
{
    Q_OBJECT
private:
    QList<int> m_vid;
    int m_pid;
    int m_boardModel;
    int m_runState;
    QList<USBPortInfo> availableDevices;
signals:
    void deviceDiscovered();
    void deviceRemoved();
private slots:
    void m_deviceDiscovered(USBPortInfo port);
    void m_deviceRemoved(USBPortInfo port);
public:
    USBSignalFilter(int vid, int pid, int boardModel, int runState);
    USBSignalFilter(QList<int> vid, int pid, int boardModel, int runState);
};
#endif // USBSIGNALFILTER_H
