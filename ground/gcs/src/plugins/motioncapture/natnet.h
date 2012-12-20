/**
 ******************************************************************************
 *
 * @file       natnet.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief
 * @see        The GNU Public License (GPL) Version 3
 * @defgroup   mocap
 * @{
 *
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

#ifndef NATNET_H
#define NATNET_H

#include <QObject>
#include "exporter.h"

#define MAX_NAMELENGTH 256

class NatNet: public Exporter
{
	Q_OBJECT
public:
    NatNet(const MocapSettings& params, Ui_MoCapWidget *widget);
    ~NatNet();
        bool setupProcess();

	void setupUdpPorts(const QString& host, int inPort, int outPort);
    void setTrackableIdx(int trackIdx);
    void setTrackableName(QString trackName);
    int getTrackableIdx();
    void setWidget(Ui_MoCapWidget *widget){this->widget=widget;}

private slots:
	void transmitUpdate();


private:
    int trackableIndex;
    bool trackUpdate;
    QString trackableName;

    void processUpdate(const QByteArray& data);
    Ui_MoCapWidget *widget;
};

class NatNetCreator : public MocapCreator
{
public:
    NatNetCreator(const QString& classId, const QString& description)
    :  MocapCreator (classId,description)
    {}

    Exporter* createExporter(const MocapSettings& params, Ui_MoCapWidget *widget)
    {
        return new NatNet(params, widget);
    }
};

#endif // NATNET_H
