#ifndef QSSPT_H
#define QSSPT_H

#include "qssp.h"
#include <QThread>
#include <QQueue>
#include <QWaitCondition>
class qsspt:public qssp, public QThread
{
public:
    qsspt(port * info,bool debug);
    void run();
    int packets_Available();
    int read_Packet(void *);
    ~qsspt();
    bool sendData(quint8 * buf,quint16 size);
private:
    virtual void pfCallBack( quint8 *, quint16);
    quint8 * mbuf;
    quint16 msize;
    QQueue<QByteArray> queue;
    QMutex mutex;
    QMutex sendbufmutex;
    bool datapending;
    bool endthread;
    quint16 sendstatus;
    quint16 receivestatus;
    QWaitCondition sendwait;
    QMutex msendwait;
    bool debug;
};

#endif // QSSPT_H
