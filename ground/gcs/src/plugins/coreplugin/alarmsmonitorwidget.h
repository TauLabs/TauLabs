/**
 ******************************************************************************
 * @file       alarmsmonitorwidget.h
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief Provides a compact summary of alarms on the tool bar
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
#ifndef ALARMSMONITORWIDGET_H
#define ALARMSMONITORWIDGET_H

#include <QObject>
#include <QGraphicsSvgItem>
#include <QGraphicsTextItem>
#include <QSvgRenderer>
#include <QTimer>

class AlarmsMonitorWidget : public QObject
{
    Q_OBJECT
public:
    static AlarmsMonitorWidget& getInstance()
    {
        static AlarmsMonitorWidget instance;
        return instance;
    }
    void init(QSvgRenderer * renderer, QGraphicsSvgItem *graph);
signals:
    
public slots:
    void processAlerts();
    void updateMessages();
    void updateNeeded();
private:
    AlarmsMonitorWidget();
    AlarmsMonitorWidget(AlarmsMonitorWidget const&);              // Don't Implement.
    void operator=(AlarmsMonitorWidget const&); // Don't implement
    QGraphicsSvgItem *error_sym;
    QGraphicsSvgItem *warning_sym;
    QGraphicsSvgItem *info_sym;
    QGraphicsTextItem *error_txt;
    QGraphicsTextItem *warning_txt;
    QGraphicsTextItem *info_txt;
    bool hasErrors;
    bool hasWarnings;
    bool hasInfos;
    QTimer alertTimer;
    bool needsUpdate;
};

#endif // ALARMSMONITORWIDGET_H
