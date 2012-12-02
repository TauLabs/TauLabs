#ifndef UAVTALKRELAY_H
#define UAVTALKRELAY_H
#include <QObject>
#include <QTcpServer>
#include <QNetworkSession>
#include <coreplugin/connectionmanager.h>
#include <QTcpSocket>
#include "uavobjectmanager.h"

class UavTalkRelay: public QObject
{
    Q_OBJECT
public:
    UavTalkRelay(UAVObjectManager * ObjMngr,QString IpAdress,int Port, bool UseTcp);
    quint16 Port(){return m_Port;}
    QString IpAdress(){return m_IpAddress;}
    bool UseTCP(){return m_UseTCP;}
    void setPort(quint16 value);
    void setIpAdress(QString value);
    void setUseTCP(bool value);
private slots:
    void newConnection();
private:
    QString m_IpAddress;
    quint16 m_Port;
    bool m_UseTCP;
    QTcpServer *tcpServer;
    QStringList fortunes;
    QNetworkSession *networkSession;
    UAVObjectManager * m_ObjMngr;
};

#endif // UAVTALKRELAY_H
