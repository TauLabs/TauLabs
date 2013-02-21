/**
 ******************************************************************************
 *
 * @file       scopegadgetwidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
 * @brief      Scope Plugin Gadget Widget
 * @see        The GNU Public License (GPL) Version 3
 * @defgroup   scopeplugin
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

#ifndef SCOPEGADGETWIDGET_H_
#define SCOPEGADGETWIDGET_H_

#include "plotdata2d.h"
#include "plotdata3d.h"

#include "qwt/src/qwt.h"
#include "qwt/src/qwt_plot_grid.h"
#include "qwt/src/qwt_plot_layout.h"


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

    void setupSeriesPlot();
    void setupTimeSeriesPlot();
    void setupHistogramPlot();
    void setupSpectrogramPlot();

    void setXWindowSize(double xWindowSize){m_xWindowSize = xWindowSize;}
    void setRefreshInterval(double refreshInterval){m_refreshInterval = refreshInterval;}
    double getXWindowSize(){return m_xWindowSize;}
    int getRefreshInterval(){return m_refreshInterval;}


    void add2dCurvePlot(QString uavObject, QString uavFieldSubField, int scaleOrderFactor = 0, int meanSamples = 1, QString mathFunction = "None", QPen pen = QPen(Qt::black));
    void addHistogram(QString uavObject, QString uavFieldSubField, double binWidth, uint numberOfBins, int scaleOrderFactor = 0, int meanSamples = 1, QString mathFunction = "None", QBrush brush = QBrush(Qt::red));
    void addWaterfallPlot(QString uavObject, QString uavFieldSubField, int scaleOrderFactor = 0, int meanSamples = 1, QString mathFunction = "None", double timeHorizon=60, double samplingFrequency=50, int windowWidth=64, double zMaximum=100);
    void clearCurvePlots();

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

    void preparePlot2d(Plot2dType plotType, Scatterplot2dType scatterplot2dType = (Scatterplot2dType) -1);
    void preparePlot3d(Plot3dType plotType);
    void setupExamplePlot();

    Plot2dType m_plot2dType;
    Plot3dType m_plot3dType;

    Scatterplot2dType m_Scatterplot2dType;

    double m_xWindowSize;
    int m_refreshInterval;
    QList<QString> m_connectedUAVObjects;
    QMap<QString, Plot2dData*> m_curves2dData;
    QMap<QString, Plot3dData*> m_curves3dData;

    static QTimer *replotTimer;

    QwtPlotGrid *m_grid;
    QwtLegend *m_legend;

	QMutex mutex;

	void deleteLegend();
	void addLegend();
};


#endif /* SCOPEGADGETWIDGET_H_ */
