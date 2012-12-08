/**
 ******************************************************************************
 * @file       uavtalkrelay.h
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
#ifndef UAVTALKRELAY_H
#define UAVTALKRELAY_H
#include <QObject>
#include <QTcpServer>
#include <QNetworkSession>
#include <coreplugin/connectionmanager.h>
#include <QTcpSocket>
#include "uavobjectmanager.h"
#include "uavtalkrelay_global.h"

class FilteredUavTalk;
class UavTalkRelay: public QObject
{
    Q_OBJECT
public:
    UavTalkRelay(UAVObjectManager * ObjMngr,QString IpAdress,quint16 Port,QHash<QString,QHash<quint32,UavTalkRelayComon::accessType> > rules,UavTalkRelayComon::accessType defaultRule);
    quint16 Port(){return m_Port;}
    QString IpAdress(){return m_IpAddress;}
    void setPort(quint16 value);
    void setIpAdress(QString value);
    void setRules(QHash<QString,QHash<quint32,UavTalkRelayComon::accessType> > value);
    void restartServer();
private slots:
    void newConnection();
private:
    QString m_IpAddress;
    quint16 m_Port;
    QTcpServer *tcpServer;
    QStringList fortunes;
    QNetworkSession *networkSession;
    UAVObjectManager * m_ObjMngr;
    QHash<QString,QHash<quint32,UavTalkRelayComon::accessType> > m_rules;
    UavTalkRelayComon::accessType m_DefaultRule;
    QList< QPointer<FilteredUavTalk> > uavTalkList;
};

#endif // UAVTALKRELAY_H
