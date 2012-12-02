#include "uavtalkrelay.h"
#include "../uavtalk/uavtalk.h"
#include "QMessageBox"

UavTalkRelay::UavTalkRelay(UAVObjectManager *ObjMngr, QString IpAdress, int Port, bool UseTcp):m_IpAddress(IpAdress),m_Port(Port),m_UseTCP(UseTcp),m_ObjMngr(ObjMngr)
{
    m_IpAddress="192.168.1.70";
    qDebug()<<"SERVER TRYING listening on "<<m_IpAddress<<m_Port;
    tcpServer = new QTcpServer(this);

    // if we did not find one, use IPv4 localhost
    if (m_IpAddress.isEmpty())
        m_IpAddress = QHostAddress(QHostAddress::LocalHost).toString();
    if (!tcpServer->listen(QHostAddress(m_IpAddress),m_Port)) {

        return;
    }
    connect(tcpServer, SIGNAL(newConnection()), this, SLOT(newConnection()));
    qDebug()<<"SERVER listening on "<<tcpServer->serverAddress()<<tcpServer->serverPort();
}

void UavTalkRelay::setPort(quint16 value)
{
}

void UavTalkRelay::setIpAdress(QString value)
{
}

void UavTalkRelay::setUseTCP(bool value)
{
}

void UavTalkRelay::newConnection()
{
    qDebug()<<"NEW CONNECTION";
      QTcpSocket *clientConnection = tcpServer->nextPendingConnection();
      connect(clientConnection, SIGNAL(disconnected()),
               clientConnection, SLOT(deleteLater()));
      UAVTalk * uav=new UAVTalk(clientConnection,m_ObjMngr);


      QList< QList<UAVObject*> > list;
      list = m_ObjMngr->getObjects();
      QList< QList<UAVObject*> >::const_iterator i;
      QList<UAVObject*>::const_iterator j;
      int objects = 0;

      for (i = list.constBegin(); i != list.constEnd(); ++i)
      {
          for (j = (*i).constBegin(); j != (*i).constEnd(); ++j)
          {
              connect(*j, SIGNAL(objectUpdated(UAVObject*)), uav, SLOT(sendObject(UAVObject*)));
              objects++;
          }
      }
  /*
      GCSTelemetryStats* gcsStatsObj = GCSTelemetryStats::GetInstance(objManager);
      GCSTelemetryStats::DataFields gcsStats = gcsStatsObj->getData();
      if ( gcsStats.Status == GCSTelemetryStats::STATUS_CONNECTED )
      {
          qDebug() << "Logging: connected already, ask for all settings";
          retrieveSettings();
      } else {
          qDebug() << "Logging: not connected, do no ask for settings";
      }*/
}
