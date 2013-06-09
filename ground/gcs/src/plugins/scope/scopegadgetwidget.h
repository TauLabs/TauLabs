/**
 ******************************************************************************
 *
 * @file       scopegadgetwidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Scope Plugin Gadget Widget
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ScopePlugin Scope Gadget Plugin
 * @{
 * @brief The scope Gadget, graphically plots the states of UAVObjects
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

#ifndef SCOPEGADGETWIDGET_H_
#define SCOPEGADGETWIDGET_H_


class ScopeConfig;
class UAVDataObject;

#include "qwt/src/qwt.h"
#include "qwt/src/qwt_plot.h"
#include "qwt/src/qwt_plot_grid.h"
#include "qwt/src/qwt_plot_layout.h"
#include "qwt/src/qwt_scale_draw.h"

#include "uavobject.h"
#include "plotdata.h"

#include <QTimer>
#include <QTime>
#include <QVector>
#include <QMutex>

/*!
  \brief This class is used to render the time values on the horizontal axis for the
  ChronoPlot.
  */
class TimeScaleDraw : public QwtScaleDraw
{
public:
    TimeScaleDraw() {
        //baseTime = QDateTime::currentDateTime().toTime_t();
    }
    virtual QwtText label(double v) const {
        uint seconds = (uint)(v);
        QDateTime upTime = QDateTime::fromTime_t(seconds);
        QTime timePart = upTime.time().addMSecs((v - seconds )* 1000);
        upTime.setTime(timePart);
        return upTime.toLocalTime().toString("hh:mm:ss");
    }
private:
//    double baseTime;
};

class ScopeGadgetWidget : public QwtPlot
{
    Q_OBJECT

public:
    ScopeGadgetWidget(QWidget *parent = 0);
    ~ScopeGadgetWidget();

    QString getUavObjectFieldUnits(QString uavObjectName, QString uavObjectFieldName);
    void connectUAVO(UAVDataObject* obj);

    void setScope(ScopeConfig *val){m_scope = val;}
    QMap<QString, PlotData*> getDataSources(){return m_dataSources;}
    void insertDataSources(QString stringVal, PlotData* dataVal){m_dataSources.insert(stringVal, dataVal);}

    void addLegend();
    void deleteLegend();
    void clearPlotWidget();
    void startTimer(int);
    QwtPlotGrid *m_grid;
    QwtLegend *m_legend;

protected:
    void mousePressEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);
    void mouseDoubleClickEvent(QMouseEvent *e);
    void mouseMoveEvent(QMouseEvent *e);
    void wheelEvent(QWheelEvent *e);
    void showEvent(QShowEvent *event);

private slots:
    void uavObjectReceived(UAVObject*);
    void replotNewData();
    void showCurve(QwtPlotItem *item, bool on);
    void startPlotting();
    void stopPlotting();

private:
    QMutex mutex;
    int m_refreshInterval;
    ScopeConfig *m_scope;
    QMap<QString, PlotData*> m_dataSources;
    double m_xWindowSize;
    static QTimer *replotTimer;
    QList<QString> m_connectedUAVObjects;

};


#endif /* SCOPEGADGETWIDGET_H_ */
