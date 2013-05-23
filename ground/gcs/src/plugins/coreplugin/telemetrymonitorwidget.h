/**
 ******************************************************************************
 * @file       telemetrymonitorwidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011-2012.
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief Provides a compact summary of telemetry on the tool bar
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
#ifndef TELEMETRYMONITORWIDGET_H
#define TELEMETRYMONITORWIDGET_H

#include <QWidget>
#include <QObject>
#include <QGraphicsView>
#include <QtSvg/QSvgRenderer>
#include <QtSvg/QGraphicsSvgItem>
#include <QtCore/QPointer>
#include <QTimer>

class TelemetryMonitorWidget : public QGraphicsView
{
    Q_OBJECT
public:
    explicit TelemetryMonitorWidget(QWidget *parent = 0);
    ~TelemetryMonitorWidget();

    void setMin(double min) { minValue = min;}
    double getMin() { return minValue; }
    void setMax(double max) { maxValue = max;}
    double getMax() { return maxValue; }

    //number of tx/rx nodes in the graph
    static const int NODE_NUMELEM = 7;
    QSvgRenderer * getRenderer(){return renderer;}
    QGraphicsSvgItem * getBackgroundItem(){return graph;}
signals:
    
public slots:
    void connected();
    void disconnect();

    void updateTelemetry(double txRate, double rxRate);
    void showTelemetry();

protected:
    void showEvent(QShowEvent *event);
    void resizeEvent(QResizeEvent *event);

private:
   QGraphicsSvgItem *graph;
   QPointer<QGraphicsTextItem> txSpeed;
   QPointer<QGraphicsTextItem> rxSpeed;
   QList<QGraphicsSvgItem*> txNodes;
   QList<QGraphicsSvgItem*> rxNodes;
   bool   m_connected;
   double txIndex;
   double txValue;
   double rxIndex;
   double rxValue;
   double minValue;
   double maxValue;
   QSvgRenderer *renderer;
};

#endif // TELEMETRYMONITORWIDGET_H
