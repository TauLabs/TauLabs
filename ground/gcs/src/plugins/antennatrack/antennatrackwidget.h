/**
 ******************************************************************************
 *
 * @file       antennatrackwidget.h
 * @author     Sami Korhonen & the OpenPilot team Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup AntennaTrackGadgetPlugin Antenna Track Gadget Plugin
 * @{
 * @brief A gadget that communicates with antenna tracker and enables basic configuration
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

#ifndef ANTENNATRACKWIDGET_H_
#define ANTENNATRACKWIDGET_H_

#include "ui_antennatrackwidget.h"
#include "antennatrackgadgetconfiguration.h"
#include "uavobject.h"
#include <QGraphicsView>
#include <QtSvg/QSvgRenderer>
#include <QtSvg/QGraphicsSvgItem>
#include <QtSerialPort/QSerialPort>
#include <QPointer>

class Ui_AntennaTrackWidget;

typedef struct struct_TrackData
{
        double Latitude;
        double Longitude;
        double Altitude;
        double HomeLatitude;
        double HomeLongitude;
        double HomeAltitude;

}TrackData_t;

class AntennaTrackWidget : public QWidget, public Ui_AntennaTrackWidget
{
    Q_OBJECT

public:
    AntennaTrackWidget(QWidget *parent = 0);
   ~AntennaTrackWidget();
   TrackData_t TrackData;
   void setPort(QPointer<QSerialPort> portx);

private slots:
   void setPosition(double, double, double);
   void setHomePosition(double, double, double);
   void dumpPacket(const QString &packet);

private:
   void calcAntennaPosition(void);
   QGraphicsSvgItem * marker;
   QPointer<QSerialPort> port;
   double azimuth_old;
   double elevation_old;
};
#endif /* ANTENNATRACKWIDGET_H_ */
