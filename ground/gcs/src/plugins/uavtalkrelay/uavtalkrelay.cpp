/**
 ******************************************************************************
 * @file       uavtalkrelay.c
 * @author     The PhoenixPilot Team, http://github.com/PhoenixPilot
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVTalk relay plugin
 * @{
 *
 * @brief Relays UAVTalk data trough UDP to another GCS
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

#include "uavtalkrelay.h"
#include "QMessageBox"
#include <QPointer>
#include "filtereduavtalk.h"

UavTalkRelay::UavTalkRelay(UAVObjectManager *ObjMngr, QString IpAdress, quint16 Port,QHash<QString,QHash<quint32,UavTalkRelayComon::accessType> > rules,UavTalkRelayComon::accessType defaultRule):m_IpAddress(IpAdress),m_Port(Port),m_ObjMngr(ObjMngr),m_rules(rules),m_DefaultRule(defaultRule)
{
    tcpServer = new QTcpServer(this);
    // if we did not find one, use IPv4 localhost
    if (m_IpAddress.isEmpty())
        m_IpAddress = QHostAddress(QHostAddress::Any).toString();
    if (!tcpServer->listen(QHostAddress(m_IpAddress),m_Port)) {

        return;
    }
    connect(tcpServer, SIGNAL(newConnection()), this, SLOT(newConnection()));
    qDebug()<<__FUNCTION__<<"SERVER listening on "<<tcpServer->serverAddress()<<tcpServer->serverPort();
}

void UavTalkRelay::setPort(quint16 value)
{
    m_Port=value;
}

void UavTalkRelay::setIpAdress(QString value)
{
    m_IpAddress=value;
}

void UavTalkRelay::setRules(QHash<QString, QHash<quint32, UavTalkRelayComon::accessType> > value)
{
    m_rules=value;
}

void UavTalkRelay::restartServer()
{
    tcpServer->close();
    if (m_IpAddress.isEmpty())
        m_IpAddress = QHostAddress(QHostAddress::Any).toString();
    if (!tcpServer->listen(QHostAddress(m_IpAddress),m_Port)) {

        return;
    }
}

void UavTalkRelay::newConnection()
{
    qDebug()<<__FUNCTION__<<"NEW CONNECTION";
    QTcpSocket *clientConnection = tcpServer->nextPendingConnection();
    connect(clientConnection, SIGNAL(disconnected()),
            clientConnection, SLOT(deleteLater()));
    qDebug()<<clientConnection->peerAddress().toString();
    QHash<quint32,UavTalkRelayComon::accessType> temp= m_rules.value(clientConnection->peerAddress().toString());
    temp.unite(m_rules.value("*"));
    QPointer<FilteredUavTalk> uav=new FilteredUavTalk(clientConnection,m_ObjMngr,temp,m_DefaultRule);
    uavTalkList.append(uav);
    connect(clientConnection, SIGNAL(disconnected()),
            uav, SLOT(deleteLater()));
    QList< QList<UAVObject*> > list;
    list = m_ObjMngr->getObjects();
    QList< QList<UAVObject*> >::const_iterator i;
    QList<UAVObject*>::const_iterator j;
    int objects = 0;
    for (i = list.constBegin(); i != list.constEnd(); ++i)
    {
        for (j = (*i).constBegin(); j != (*i).constEnd(); ++j)
        {
            connect(*j, SIGNAL(objectUpdated(UAVObject*)), uav.data(), SLOT(sendObjectSlot(UAVObject*)));
            objects++;
        }
    }
}
