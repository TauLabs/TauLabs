#include "port.h"
#include "delay.h"
port::port(PortSettings settings,QString name):mstatus(port::closed)
{
    timer.start();
    sport = new QextSerialPort(name,settings, QextSerialPort::Polling);
    if(sport->open(QIODevice::ReadWrite|QIODevice::Unbuffered))
    {
        mstatus=port::open;
      //  sport->setDtr();
    }
    else
        mstatus=port::error;
}
port::portstatus port::status()
{
    return mstatus;
}
qint16 port::pfSerialRead(void)
{

    char c[1];
    if(sport->bytesAvailable())
    {
        sport->read(c,1);
    }
    else return -1;
    return (quint8)c[0];
}

void port::pfSerialWrite(quint8 c)
{
    char cc[1];
    cc[0]=c;
    sport->write(cc,1);
}

quint32 port::pfGetTime(void)
{
    return timer.elapsed();
}
