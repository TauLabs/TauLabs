/**
 ******************************************************************************
 *
 * @file       scopegadgetwidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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


#include <QDir>

#include "scopegadgetwidget.h"
#include "scopegadgetconfiguration.h"

#include "utils/stylehelper.h"
#include "uavtalk/telemetrymanager.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include "coreplugin/icore.h"
#include "coreplugin/connectionmanager.h"

#include "qwt/src/qwt_legend.h"
#include "qwt/src/qwt_legend_item.h"
#include "qwt/src/qwt_double_range.h"
#include "qwt/src/qwt_scale_widget.h"

#include <iostream>
#include <math.h>
#include <QDebug>
#include <QColor>
#include <QStringList>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QMutexLocker>
#include <QWheelEvent>


QTimer *ScopeGadgetWidget::replotTimer=0;

ScopeGadgetWidget::ScopeGadgetWidget(QWidget *parent) : QwtPlot(parent),
    m_refreshInterval(50), // Arbitrary 50ms refresh timer
    m_scope(0),
    m_xWindowSize(60) // This is an arbitrary 1 minute window
{
    m_grid = new QwtPlotGrid;

    setMouseTracking(true);
//	canvas()->setMouseTracking(true);

    //Set up the timer that replots data. Only set up one timer for entire class.
    if (replotTimer == NULL)
        replotTimer = new QTimer();
    connect(replotTimer, SIGNAL(timeout()), this, SLOT(replotNewData()));

    // Listen to telemetry connection/disconnection events, no point in
    // running the scopes if we are not connected and not replaying logs.
    // Also listen to disconnect actions from the user
    Core::ConnectionManager *cm = Core::ICore::instance()->connectionManager();
    connect(cm, SIGNAL(deviceAboutToDisconnect()), this, SLOT(stopPlotting()));
    connect(cm, SIGNAL(deviceConnected(QIODevice*)), this, SLOT(startPlotting()));
}

/**
 * @brief ScopeGadgetWidget::~ScopeGadgetWidget Destructor
 */
ScopeGadgetWidget::~ScopeGadgetWidget()
{
    // Get the object to de-monitor
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    foreach (QString uavObjName, m_connectedUAVObjects)
    {
        UAVDataObject *obj = dynamic_cast<UAVDataObject*>(objManager->getObject(uavObjName));
        disconnect(obj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(uavObjectReceived(UAVObject*)));
    }

    // Clear the plot
    clearPlotWidget();
}

// ******************************************************************

/**
 * @brief ScopeGadgetWidget::mousePressEvent Pass mouse press event to QwtPlot
 * @param e
 */
void ScopeGadgetWidget::mousePressEvent(QMouseEvent *e)
{
    QwtPlot::mousePressEvent(e);
}


/**
 * @brief ScopeGadgetWidget::mouseReleaseEvent Pass mouse release event to QwtPlot
 * @param e
 */
void ScopeGadgetWidget::mouseReleaseEvent(QMouseEvent *e)
{
    QwtPlot::mouseReleaseEvent(e);
}


/**
 * @brief ScopeGadgetWidget::mouseDoubleClickEvent Turn legend on and off, then pass double-click even to QwtPlot
 * @param e
 */
void ScopeGadgetWidget::mouseDoubleClickEvent(QMouseEvent *e)
{
    //On double-click, toggle legend
    mutex.lock();

    if (legend())
        deleteLegend();
    else
        addLegend();

    mutex.unlock();

    //On double-click, reset plot zoom
    setAxisAutoScale(QwtPlot::yLeft, true);

    update();

    QwtPlot::mouseDoubleClickEvent(e);
}


/**
 * @brief ScopeGadgetWidget::mouseMoveEvent Pass mouse move event to QwtPlot
 * @param e
 */
void ScopeGadgetWidget::mouseMoveEvent(QMouseEvent *e)
{
    QwtPlot::mouseMoveEvent(e);
}


/**
 * @brief ScopeGadgetWidget::wheelEvent Zoom in or out, then pass mouse wheel event to QwtPlot
 * @param e
 */
void ScopeGadgetWidget::wheelEvent(QWheelEvent *e)
{
    //Change zoom on scroll wheel event
    QwtInterval yInterval=axisInterval(QwtPlot::yLeft);
    if (yInterval.minValue() != yInterval.maxValue()) //Make sure that the two values are never the same. Sometimes axisInterval returns (0,0)
    {
        //Determine what y value to zoom about. NOTE, this approach has a bug that the in that
        //the value returned by Qt includes the legend, whereas the value transformed by Qwt
        //does *not*. Thus, when zooming with a legend, there will always be a small bias error.
        //In practice, this seems not to be a UI problem.
        QPoint mouse_pos=e->pos(); //Get the mouse coordinate in the frame
        double zoomLine=invTransform(QwtPlot::yLeft, mouse_pos.y()); //Transform the y mouse coordinate into a frame value.

        double zoomScale=1.1; //THIS IS AN ARBITRARY CONSTANT, AND PERHAPS SHOULD BE IN A DEFINE INSTEAD OF BURIED HERE

        mutex.lock(); //DOES THIS mutex.lock NEED TO BE HERE? I DON'T KNOW, I JUST COPIED IT FROM THE ABOVE CODE
        // Set the scale
        if (e->delta()<0){
            setAxisScale(QwtPlot::yLeft,
                         (yInterval.minValue()-zoomLine)*zoomScale+zoomLine,
                         (yInterval.maxValue()-zoomLine)*zoomScale+zoomLine );
        }
        else{
            setAxisScale(QwtPlot::yLeft,
                         (yInterval.minValue()-zoomLine)/zoomScale+zoomLine,
                         (yInterval.maxValue()-zoomLine)/zoomScale+zoomLine );
        }
        mutex.unlock();
    }
    QwtPlot::wheelEvent(e);
}


/**
 * @brief ScopeGadgetWidget::startPlotting Starts/stops telemetry
 */
void ScopeGadgetWidget::startPlotting()
{
    if (!replotTimer)
        return;

    if (!replotTimer->isActive())
        replotTimer->start(m_refreshInterval);
}


/**
 * @brief ScopeGadgetWidget::stopPlotting Stops plotting timer
 */
void ScopeGadgetWidget::stopPlotting()
{
    if (!replotTimer)
        return;

    replotTimer->stop();
}


/**
 * @brief ScopeGadgetWidget::deleteLegend Delete legend from plot
 */
void ScopeGadgetWidget::deleteLegend()
{
    if (!legend())
        return;

    disconnect(this, SIGNAL(legendChecked(QwtPlotItem *, bool)), this, 0);

    m_legend->clear();
    insertLegend(NULL, QwtPlot::TopLegend);
}


/**
 * @brief ScopeGadgetWidget::addLegend Add legend to plot
 */
void ScopeGadgetWidget::addLegend()
{
    if (legend())
        return;

    // Show a legend at the top
    m_legend = new QwtLegend;
    m_legend->setItemMode(QwtLegend::CheckableItem);
    m_legend->setFrameStyle(QFrame::Box | QFrame::Sunken);
    m_legend->setToolTip(tr("Click legend to show/hide scope trace"));

    QPalette pal = m_legend->palette();
    pal.setColor(m_legend->backgroundRole(), QColor(100, 100, 100));	// background colour
    pal.setColor(QPalette::Text, QColor(0, 0, 0));			// text colour
    m_legend->setPalette(pal);

    insertLegend(m_legend, QwtPlot::TopLegend);

    // Update the checked/unchecked state of the legend items
    // -> this is necessary when hiding a legend where some plots are
    //    not visible, and the un-hiding it.
    foreach (QwtPlotItem *item, this->itemList()) {
        bool on = item->isVisible();
        QWidget *w = m_legend->find(item);
        if ( w && w->inherits("QwtLegendItem") )
            ((QwtLegendItem *)w)->setChecked(!on);
    }

    connect(this, SIGNAL(legendChecked(QwtPlotItem *, bool)), this, SLOT(showCurve(QwtPlotItem *, bool)));
}


/**
 * @brief ScopeGadgetWidget::showCurve
 * @param item
 * @param on
 */
void ScopeGadgetWidget::showCurve(QwtPlotItem *item, bool on)
{
    item->setVisible(!on);
    QWidget *w = legend()->find(item);
    if ( w && w->inherits("QwtLegendItem") )
        ((QwtLegendItem *)w)->setChecked(on);

    mutex.lock();
        replot();
    mutex.unlock();
}


/**
 * @brief ScopeGadgetWidget::uavObjectReceived
 * @param obj
 */
void ScopeGadgetWidget::uavObjectReceived(UAVObject* obj)
{
    foreach(PlotData* plotdData, m_dataSources.values()) {
        bool ret = plotdData->append(obj);
        if (ret)
            plotdData->setUpdatedFlagToTrue();
    }
}



/**
 * @brief ScopeGadgetWidget::replotNewData
 */
void ScopeGadgetWidget::replotNewData()
{
    // If the plot is not visible or there is no scope, do not replot
    if (!isVisible() || m_scope == NULL)
        return;

    QMutexLocker locker(&mutex);

    // Update the data in the scopes
    foreach(PlotData* plotData, m_dataSources.values())
    {
        plotData->plotNewData(plotData, m_scope, this);
    }

    // Repaint the scopes
    replot();
}


/**
 * @brief ScopeGadgetWidget::clearPlotWidget
 */
void ScopeGadgetWidget::clearPlotWidget()
{
    if(m_grid){
        m_grid->detach();
    }
    if(m_scope){
        // Clear the plots
        foreach(PlotData* plotData, m_dataSources.values()) {
            plotData->clearPlots(plotData);
        }

        // Clear the data
        m_dataSources.clear();
    }
}


/**
 * @brief ScopeGadgetWidget::getUavObjectFieldUnits Gets the UAVOs units, as defined in the XML
 * @param uavObjectName
 * @param uavObjectFieldName
 * @return
 */
QString ScopeGadgetWidget::getUavObjectFieldUnits(QString uavObjectName, QString uavObjectFieldName)
{
    //Get the uav object
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* obj = dynamic_cast<UAVDataObject*>(objManager->getObject(uavObjectName));
    if(!obj) {
        qDebug() << "In scope gadget, UAVObject " << uavObjectName << " is missing";
        return "";
    }
    UAVObjectField* field = obj->getField(uavObjectFieldName);
    if(!field) {
        qDebug() << "In scope gadget, in fields loaded from GCS config file, field" << uavObjectFieldName << " of UAVObject " << uavObjectName << " is missing";
        return "";
    }

    //Get the units
    QString units = field->getUnits();
    if(units == 0)
        units = QString();

    return units;
}


/**
 * @brief ScopeGadgetWidget::showEvent Reimplemented from QwtPlot
 * @param event
 */
void ScopeGadgetWidget::showEvent(QShowEvent *event)
{
    replotNewData();
    QwtPlot::showEvent(event);
}


/**
 * @brief ScopeGadgetWidget::connectUAVO Connects UAVO update signal, but only if it hasn't yet been connected
 * @param obj
 */
void ScopeGadgetWidget::connectUAVO(UAVDataObject* obj){
    //Link to the new signal data only if this UAVObject has not been connected yet
    if (!m_connectedUAVObjects.contains(obj->getName())) {
        m_connectedUAVObjects.append(obj->getName());
        connect(obj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(uavObjectReceived(UAVObject*)));
    }

}


/**
 * @brief ScopeGadgetWidget::startTimer Starts timer
 * @param refreshInterval
 */
void ScopeGadgetWidget::startTimer(int refreshInterval){
    m_refreshInterval = refreshInterval;

    // Only start the timer if we are already connected
    Core::ConnectionManager *cm = Core::ICore::instance()->connectionManager();
    if (cm->getCurrentConnection() && replotTimer)
    {
        if (!replotTimer->isActive())
            replotTimer->start(refreshInterval);
        else
            replotTimer->setInterval(refreshInterval);

    }
}
