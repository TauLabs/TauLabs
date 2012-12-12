/**
 ******************************************************************************
 * @file       filtereduavtalk.h
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
#ifndef FILTEREDUAVTALK_H
#define FILTEREDUAVTALK_H

#include "../uavtalk/uavtalk.h"
#include <QHash>
#include "uavtalkrelay_global.h"

/**
 * @brief The FilteredUavTalk class An extension of the UAVTalk class to be run on the master
 * GCS (the one which also has a connection to the UAV) which will relay object updates to
 * a slave GCS subject to certain filtering rules which this class enforces.
 */
class UAVTALKRELAY_EXPORT FilteredUavTalk:public UAVTalk
{
    Q_OBJECT
public:
    FilteredUavTalk(QIODevice* iodev, UAVObjectManager* objMngr,QHash<quint32,UavTalkRelayComon::accessType> rules,UavTalkRelayComon::accessType defaultRule);

    //! Called when an uavtalk packet is received from the slave.  Updates master based on filtering rules
    bool receiveObject(quint8 type, quint32 objId, quint16 instId, quint8* data, qint32 length);

public slots:
    //! Called whenever an object is updated either locally in the master GCS or from the main
    //! telemetry connection, but NOT as a consquence of the receiveObject method
    void sendObjectSlot(UAVObject *obj);

private:
    QHash<quint32,UavTalkRelayComon::accessType> m_rules;
    UavTalkRelayComon::accessType m_defaultRule;
};

#endif // FILTEREDUAVTALK_H
