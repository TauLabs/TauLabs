/**
 ******************************************************************************
 *
 * @file       scopegadgetwidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
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
#include "scopes2d/histogramdata.h"
#include "scopes2d/scatterplotdata.h"
#include "scopes3d/spectrogramdata.h"
#include "scopegadgetwidget.h"
#include "utils/stylehelper.h"

#include "scopegadgetconfiguration.h"

#include "uavtalk/telemetrymanager.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include "coreplugin/icore.h"
#include "coreplugin/connectionmanager.h"

#include "qwt/src/qwt_legend.h"
#include "qwt/src/qwt_legend_item.h"
#include "qwt/src/qwt_double_range.h"


#include <iostream>
#include <math.h>
#include <QDebug>
#include <QColor>
#include <QStringList>
#include <QtGui/QWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QPushButton>
#include <QMutexLocker>
#include <QWheelEvent>


QTimer *ScopeGadgetWidget::replotTimer=0;

ScopeGadgetWidget::ScopeGadgetWidget(QWidget *parent) : QwtPlot(parent),
  m_plot2dType(NO2DPLOT),
  m_plot3dType(NO3DPLOT)
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

    clearCurvePlots();
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
 * @brief ScopeGadgetWidget::preparePlot2d Prepare plot background, color, etc...
 * @param plotType Type of plot as supported by GCS.
 */
void ScopeGadgetWidget::preparePlot2d(Plot2dType plotType, Scatterplot2dType scatterplot2dType)
{
    m_plot2dType = plotType;

    clearCurvePlots();

    setMinimumSize(64, 64);
    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    setCanvasBackground(QColor(64, 64, 64));

    switch(m_plot2dType)
    {
    case HISTOGRAM:
        plotLayout()->setAlignCanvasToScales( false );

        m_grid->enableX( false );
        m_grid->enableY( true );
        m_grid->enableXMin( false );
        m_grid->enableYMin( false );
        m_grid->setMajPen( QPen( Qt::black, 0, Qt::DotLine ) );
        m_grid->setMinPen(QPen(Qt::lightGray, 0, Qt::DotLine));
        m_grid->setPen(QPen(Qt::darkGray, 1, Qt::DotLine));
        m_grid->attach( this );

        break;
    case SCATTERPLOT2D:
        m_Scatterplot2dType = scatterplot2dType;

        //Add grid lines
        m_grid->enableX( true );
        m_grid->enableY( true );
        m_grid->enableXMin( false );
        m_grid->enableYMin( false );
        m_grid->setMajPen(QPen(Qt::gray, 0, Qt::DashLine));
        m_grid->setMinPen(QPen(Qt::lightGray, 0, Qt::DotLine));
        m_grid->setPen(QPen(Qt::darkGray, 1, Qt::DotLine));
        m_grid->attach(this);

        break;
    default:
        // We shouldn't be able to get this far
        Q_ASSERT(0);
    }

    // Add the legend
    addLegend();

    // Only start the timer if we are already connected
    Core::ConnectionManager *cm = Core::ICore::instance()->connectionManager();
    if (cm->getCurrentConnection() && replotTimer)
    {
        if (!replotTimer->isActive())
            replotTimer->start(m_refreshInterval);
        else
            replotTimer->setInterval(m_refreshInterval);
    }

}


/**
 * @brief ScopeGadgetWidget::preparePlot3d Prepare plot background, color, etc...
 * @param plotType Type of plot as supported by GCS.
 */
void ScopeGadgetWidget::preparePlot3d(Plot3dType plotType)
{
    m_plot3dType = plotType;

    clearCurvePlots();

    setMinimumSize(64, 64);
    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    setCanvasBackground(QColor(64, 64, 64));

    switch(m_plot3dType)
    {
    case SPECTROGRAM:
        //Remove grid lines
        m_grid->enableX( false );
        m_grid->enableY( false );
        m_grid->enableXMin( false );
        m_grid->enableYMin( false );
        m_grid->setMajPen(QPen(Qt::gray, 0, Qt::DashLine));
        m_grid->setMinPen(QPen(Qt::lightGray, 0, Qt::DotLine));
        m_grid->setPen(QPen(Qt::darkGray, 1, Qt::DotLine));
        m_grid->attach(this);

        break;
    case SCATTERPLOT3D:
        //Add grid lines
        m_grid->enableX( true );
        m_grid->enableY( true );
        m_grid->enableXMin( false );
        m_grid->enableYMin( false );
        m_grid->setMajPen(QPen(Qt::gray, 0, Qt::DashLine));
        m_grid->setMinPen(QPen(Qt::lightGray, 0, Qt::DotLine));
        m_grid->setPen(QPen(Qt::darkGray, 1, Qt::DotLine));
        m_grid->attach(this);

        // Add the legend
        addLegend();
        break;
    default:
        //Shouldn't be able to get here
        Q_ASSERT(0);
    }

    // Only start the timer if we are already connected
    Core::ConnectionManager *cm = Core::ICore::instance()->connectionManager();
    if (cm->getCurrentConnection() && replotTimer)
    {
        if (!replotTimer->isActive())
            replotTimer->start(m_refreshInterval);
        else
            replotTimer->setInterval(m_refreshInterval);
    }
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
 * @brief ScopeGadgetWidget::setupSeriesPlot
 */
void ScopeGadgetWidget::setupSeriesPlot()
{
    preparePlot2d(SCATTERPLOT2D, SERIES2D);

//	QwtText title("Index");
////	title.setFont(QFont("Helvetica", 20));
//	title.font().setPointSize(title.font().pointSize() / 2);
//	setAxisTitle(QwtPlot::xBottom, title);
////    setAxisTitle(QwtPlot::xBottom, "Index");

    setAxisScaleDraw(QwtPlot::xBottom, new QwtScaleDraw());
    setAxisScale(QwtPlot::xBottom, 0, m_xWindowSize);
    setAxisAutoScale(QwtPlot::yLeft, true);
    setAxisLabelRotation(QwtPlot::xBottom, 0.0);
    setAxisLabelAlignment(QwtPlot::xBottom, Qt::AlignLeft | Qt::AlignBottom);

    QwtScaleWidget *scaleWidget = axisWidget(QwtPlot::xBottom);

    // reduce the gap between the scope canvas and the axis scale
    scaleWidget->setMargin(0);

    // reduce the axis font size
    QFont fnt(axisFont(QwtPlot::xBottom));
    fnt.setPointSize(7);
    setAxisFont(QwtPlot::xBottom, fnt);	// x-axis
    setAxisFont(QwtPlot::yLeft, fnt);	// y-axis
    setAxisFont(QwtPlot::yRight, fnt);	// y-axis
}


/**
 * @brief ScopeGadgetWidget::setupTimeSeriesPlot
 */
void ScopeGadgetWidget::setupTimeSeriesPlot()
{
    preparePlot2d(SCATTERPLOT2D, TIMESERIES2D);

//    QwtText title("Time [h:m:s]");
////	title.setFont(QFont("Helvetica", 20));
//    title.font().setPointSize(title.font().pointSize() / 2);
//    setAxisTitle(QwtPlot::xBottom, title);
////	setAxisTitle(QwtPlot::xBottom, "Time [h:m:s]");

    setAxisScaleDraw(QwtPlot::xBottom, new TimeScaleDraw());
    uint NOW = QDateTime::currentDateTime().toTime_t();
    setAxisScale(QwtPlot::xBottom, NOW - m_xWindowSize / 1000, NOW);
//	setAxisLabelRotation(QwtPlot::xBottom, -15.0);
    setAxisLabelRotation(QwtPlot::xBottom, 0.0);
    setAxisLabelAlignment(QwtPlot::xBottom, Qt::AlignLeft | Qt::AlignBottom);
//	setAxisLabelAlignment(QwtPlot::xBottom, Qt::AlignCenter | Qt::AlignBottom);

    QwtScaleWidget *scaleWidget = axisWidget(QwtPlot::xBottom);
//	QwtScaleDraw *scaleDraw = axisScaleDraw();

    // reduce the gap between the scope canvas and the axis scale
    scaleWidget->setMargin(0);

    // reduce the axis font size
    QFont fnt(axisFont(QwtPlot::xBottom));
    fnt.setPointSize(7);
    setAxisFont(QwtPlot::xBottom, fnt);	// x-axis
    setAxisFont(QwtPlot::yLeft, fnt);	// y-axis
    setAxisFont(QwtPlot::yRight, fnt);	// y-axis

    // set the axis colours .. can't seem to change the background colour :(
//	QPalette pal = scaleWidget->palette();
//	QPalette::ColorRole cr = scaleWidget->backgroundRole();
//	pal.setColor(cr, QColor(128, 128, 128));				// background colour
//	cr = scaleWidget->foregroundRole();
//	pal.setColor(cr, QColor(255, 255, 255));				// tick colour
//	pal.setColor(QPalette::Text, QColor(255, 255, 255));	// text colour
//	scaleWidget->setPalette(pal);

    /*
     In situations, when there is a label at the most right position of the
     scale, additional space is needed to display the overlapping part
     of the label would be taken by reducing the width of scale and canvas.
     To avoid this "jumping canvas" effect, we add a permanent margin.
     We don't need to do the same for the left border, because there
     is enough space for the overlapping label below the left scale.
     */

//	const int fmh = QFontMetrics(scaleWidget->font()).height();
//	scaleWidget->setMinBorderDist(0, fmh / 2);

//	const int fmw = QFontMetrics(scaleWidget->font()).width(" 00:00:00 ");
//	const int fmw = QFontMetrics(scaleWidget->font()).width(" ");
//	scaleWidget->setMinBorderDist(0, fmw);
}

void ScopeGadgetWidget::setupHistogramPlot(){

        preparePlot2d(HISTOGRAM);

        setAxisScaleDraw(QwtPlot::xBottom, new QwtScaleDraw());
        setAxisAutoScale(QwtPlot::xBottom);
        setAxisLabelRotation(QwtPlot::xBottom, 0.0);
        setAxisLabelAlignment(QwtPlot::xBottom, Qt::AlignLeft | Qt::AlignBottom);

        QwtScaleWidget *scaleWidget = axisWidget(QwtPlot::xBottom);

        // reduce the gap between the scope canvas and the axis scale
        scaleWidget->setMargin(0);

        // reduce the axis font size
        QFont fnt(axisFont(QwtPlot::xBottom));
        fnt.setPointSize(7);
        setAxisFont(QwtPlot::xBottom, fnt);	// x-axis
        setAxisFont(QwtPlot::yLeft, fnt);	// y-axis

}


void ScopeGadgetWidget::setupSpectrogramPlot()
{
        preparePlot3d(SPECTROGRAM);

        // Spectrograms use autoscale
        setAxisAutoScale(QwtPlot::xBottom);
        setAxisAutoScale(QwtPlot::yLeft);
}


/**
 * @brief ScopeGadgetWidget::addWaterfallPlot Adds a waterfall-style spectrogram
 * @param uavObjectName
 * @param uavFieldSubFieldName
 * @param scaleOrderFactor
 * @param meanSamples
 * @param mathFunction
 * @param samplingFrequency
 * @param windowWidth
 * @param timeHorizon
 */
void ScopeGadgetWidget::addWaterfallPlot(QString uavObjectName, QString uavFieldSubFieldName, int scaleOrderFactor, int meanSamples,
                                         QString mathFunction, double timeHorizon, double samplingFrequency, int windowWidth, double zMaximum)
{
    SpectrogramData* spectrogramData = new SpectrogramData(uavObjectName, uavFieldSubFieldName, samplingFrequency, windowWidth, timeHorizon);

    spectrogramData->setXMinimum(0);
    spectrogramData->setXMaximum(samplingFrequency/2);
    spectrogramData->setYMinimum(0);
    spectrogramData->setYMaximum(timeHorizon);
    spectrogramData->setZMaximum(zMaximum);
    spectrogramData->setScalePower(scaleOrderFactor);
    spectrogramData->setMeanSamples(meanSamples);
    spectrogramData->setMathFunction(mathFunction);

    //Generate the waterfall name
    QString waterfallName = (spectrogramData->getUavoName()) + "." + (spectrogramData->getUavoFieldName());
    if(spectrogramData->getHaveSubFieldFlag())
        waterfallName = waterfallName.append("." + spectrogramData->getUavoSubFieldName());

    //Get the uav object
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* obj = dynamic_cast<UAVDataObject*>(objManager->getObject((spectrogramData->getUavoName())));
    if(!obj) {
        qDebug() << "Object " << spectrogramData->getUavoName() << " is missing";
        return;
    }

    //Get the units
    QString units = ScopeGadgetWidget::getUavObjectFieldUnits(spectrogramData->getUavoName(), spectrogramData->getUavoFieldName());

    //Generate name with scaling factor appeneded
    QString waterfallNameScaled;
    if(scaleOrderFactor == 0)
        waterfallNameScaled = waterfallName + "(" + units + ")";
    else
        waterfallNameScaled = waterfallName + "(x10^" + QString::number(scaleOrderFactor) + " " + units + ")";

    //Create the waterfall plot
    QwtPlotSpectrogram* plotSpectrogram = new QwtPlotSpectrogram(waterfallNameScaled);
    plotSpectrogram->setRenderThreadCount( 0 ); // use system specific thread count
    plotSpectrogram->setRenderHint(QwtPlotItem::RenderAntialiased);
    plotSpectrogram->setColorMap(new ColorMap() );

    // Initial raster data
    spectrogramData->rasterData = new QwtMatrixRasterData();

    QDateTime NOW = QDateTime::currentDateTime(); //TODO: This should show UAVO time and not system time
    for ( uint i = 0; i < timeHorizon; i++ ){
        spectrogramData->timeDataHistory->append(NOW.toTime_t() + NOW.time().msec() / 1000.0 + i);
    }

    if (((double) windowWidth) * timeHorizon < (double) 10000000.0 * sizeof(spectrogramData->zDataHistory->front())){ //Don't exceed 10MB for memory
        for ( uint i = 0; i < windowWidth*timeHorizon; i++ ){
            spectrogramData->zDataHistory->append(0);
        }
    }
    else{
        qDebug() << "For some reason, we're trying to allocate a gigantic spectrogram. This probably represents a problem in the configuration file. TimeHorizion: "<< timeHorizon << ", windowWidth: "<< windowWidth;
        Q_ASSERT(0);
        return;
    }

    int numColumns = windowWidth;
    spectrogramData->rasterData->setValueMatrix( *(spectrogramData->zDataHistory), numColumns );

    //Set the ranges for the plot
    spectrogramData->rasterData->setInterval( Qt::XAxis, QwtInterval(spectrogramData->getXMinimum(), spectrogramData->getXMaximum()));
    spectrogramData->rasterData->setInterval( Qt::YAxis, QwtInterval(spectrogramData->getYMinimum(), spectrogramData->getYMaximum()));
    spectrogramData->rasterData->setInterval( Qt::ZAxis, QwtInterval(0, zMaximum));

    //Set up colorbar on right axis
    spectrogramData->rightAxis = axisWidget( QwtPlot::yRight );
    spectrogramData->rightAxis->setTitle( "Intensity" );
    spectrogramData->rightAxis->setColorBarEnabled( true );
    spectrogramData->rightAxis->setColorMap( QwtInterval(0, zMaximum), new ColorMap() );
    setAxisScale( QwtPlot::yRight, 0, zMaximum);
    enableAxis( QwtPlot::yRight );



    plotSpectrogram->setData(spectrogramData->rasterData);

    plotSpectrogram->attach(this);
    spectrogramData->spectrogram = plotSpectrogram;

    //Keep the curve details for later
    m_curves3dData.insert(waterfallNameScaled, spectrogramData);

    //Link to the new signal data only if this UAVObject has not been connected yet
    if (!m_connectedUAVObjects.contains(obj->getName())) {
        m_connectedUAVObjects.append(obj->getName());
        connect(obj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(uavObjectReceived(UAVObject*)));
    }

    mutex.lock();
        replot();
    mutex.unlock();
}

/**
 * @brief ScopeGadgetWidget::add2dCurvePlot
 * @param uavObject
 * @param uavFieldSubField
 * @param scaleOrderFactor
 * @param meanSamples
 * @param mathFunction
 * @param pen
 */
void ScopeGadgetWidget::add2dCurvePlot(QString uavObjectName, QString uavFieldSubFieldName, int scaleOrderFactor, int meanSamples, QString mathFunction, QPen pen)
{
    ScatterplotData* scatterplotData;

    switch(m_Scatterplot2dType){
    case SERIES2D:
        scatterplotData = new SeriesPlotData(uavObjectName, uavFieldSubFieldName);
        break;
    case TIMESERIES2D:
        scatterplotData = new TimeSeriesPlotData(uavObjectName, uavFieldSubFieldName);
        break;
    }

    scatterplotData->setXWindowSize(m_xWindowSize);
    scatterplotData->setScalePower(scaleOrderFactor);
    scatterplotData->setMeanSamples(meanSamples);
    scatterplotData->setMathFunction(mathFunction);

    //If the y-bounds are provided, set them
    if (scatterplotData->getYMinimum() != scatterplotData->getYMaximum())
	{
//        setAxisScale(QwtPlot::yLeft, scatterplotData->getYMinimum(), scatterplotData->getYMaximum());
	}

    //Generate the curve name
    QString curveName = (scatterplotData->getUavoName()) + "." + (scatterplotData->getUavoFieldName());
    if(scatterplotData->getHaveSubFieldFlag())
        curveName = curveName.append("." + scatterplotData->getUavoSubFieldName());

    //Get the uav object
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* obj = dynamic_cast<UAVDataObject*>(objManager->getObject((scatterplotData->getUavoName())));
    if(!obj) {
        qDebug() << "Object " << scatterplotData->getUavoName() << " is missing";
        return;
    }

    //Get the units
    QString units = ScopeGadgetWidget::getUavObjectFieldUnits(scatterplotData->getUavoName(), scatterplotData->getUavoFieldName());

    //Generate name with scaling factor appeneded
    QString curveNameScaled;
    if(scaleOrderFactor == 0)
        curveNameScaled = curveName + "(" + units + ")";
    else
        curveNameScaled = curveName + "(x10^" + QString::number(scaleOrderFactor) + " " + units + ")";

    QString curveNameScaledMath;
    if (mathFunction == "None")
        curveNameScaledMath = curveNameScaled;
    else if (mathFunction == "Boxcar average"){
        curveNameScaledMath = curveNameScaled + " (avg)";
    }
    else if (mathFunction == "Standard deviation"){
        curveNameScaledMath = curveNameScaled + " (std)";
    }
    else
    {
        //Shouldn't be able to get here. Perhaps a new math function was added without
        // updating this list?
        Q_ASSERT(0);
    }

    //Create the curve plot
    QwtPlotCurve* plotCurve = new QwtPlotCurve(curveNameScaledMath);
    plotCurve->setPen(pen);
    plotCurve->setSamples(*(scatterplotData->getXData()), *(scatterplotData->getYData()));
    plotCurve->attach(this);
    scatterplotData->curve = plotCurve;

    //Keep the curve details for later
    m_curves2dData.insert(curveNameScaledMath, scatterplotData);

    //Link to the new signal data only if this UAVObject has not been connected yet
    if (!m_connectedUAVObjects.contains(obj->getName())) {
        m_connectedUAVObjects.append(obj->getName());
        connect(obj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(uavObjectReceived(UAVObject*)));
    }

    mutex.lock();
    replot();
    mutex.unlock();
}

void ScopeGadgetWidget::addHistogram(QString uavObjectName, QString uavFieldSubFieldName, double binWidth, uint numberOfBins, int scaleOrderFactor, int meanSamples, QString mathFunction, QBrush brush)
{
    HistogramData* histogramData;
    histogramData = new HistogramData(uavObjectName, uavFieldSubFieldName, binWidth, numberOfBins);

    histogramData->setScalePower(scaleOrderFactor);
    histogramData->setMeanSamples(meanSamples);
    histogramData->setMathFunction(mathFunction);

    //If the y-bounds are provided, set them
    if (histogramData->getYMinimum() != histogramData->getYMaximum())
    {
//        setAxisScale(QwtPlot::yLeft, histogramData->getYMinimum(), histogramData->getYMaximum());
    }

    //Generate the curve name
    QString curveName = (histogramData->getUavoName()) + "." + (histogramData->getUavoFieldName());
    if(histogramData->getHaveSubFieldFlag())
        curveName = curveName.append("." + histogramData->getUavoSubFieldName());

    //Get the uav object
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* obj = dynamic_cast<UAVDataObject*>(objManager->getObject((histogramData->getUavoName())));
    if(!obj) {
        qDebug() << "Object " << histogramData->getUavoName() << " is missing";
        return;
    }

    //Get the units
    QString units = ScopeGadgetWidget::getUavObjectFieldUnits(histogramData->getUavoName(), histogramData->getUavoFieldName());

    //Generate name with scaling factor appeneded
    QString histogramNameScaled;
    if(scaleOrderFactor == 0)
        histogramNameScaled = curveName + "(" + units + ")";
    else
        histogramNameScaled = curveName + "(x10^" + QString::number(scaleOrderFactor) + " " + units + ")";

    //Create histogram data set
    histogramData->histogramBins = new QVector<QwtIntervalSample>();
    histogramData->histogramInterval = new QVector<QwtInterval>();

    // Generate the interval series
    histogramData->intervalSeriesData = new QwtIntervalSeriesData(*histogramData->histogramBins);

    // Create the histogram
    QwtPlotHistogram* plotHistogram = new QwtPlotHistogram(histogramNameScaled);
    plotHistogram->setStyle( QwtPlotHistogram::Columns );
    plotHistogram->setBrush(brush);
    plotHistogram->setData( histogramData->intervalSeriesData);

    plotHistogram->attach(this);
    histogramData->histogram = plotHistogram;

    //Keep the curve details for later
    m_curves2dData.insert(histogramNameScaled, histogramData);

    //Link to the new signal data only if this UAVObject has not been connected yet
    if (!m_connectedUAVObjects.contains(obj->getName())) {
        m_connectedUAVObjects.append(obj->getName());
        connect(obj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(uavObjectReceived(UAVObject*)));
    }

    mutex.lock();
    replot();
    mutex.unlock();

}
//void ScopeGadgetWidget::removeCurvePlot(QString uavObjectName, QString uavFieldName)
//{
//    QString curveName = uavObjectName + "." + uavFieldName;
//
//    Plot2dData* plot2dData = m_curves2dData.take(curveName);
//    m_curves2dData.remove(curveName);
//    plot2dData->curve->detach();
//
//    delete plot2dData->curve;
//    delete plot2dData;
//
//	mutex.lock();
//  replot();
//	mutex.unlock();
//}


/**
 * @brief ScopeGadgetWidget::uavObjectReceived
 * @param obj
 */
void ScopeGadgetWidget::uavObjectReceived(UAVObject* obj)
{
    foreach(Plot2dData* plot2dData, m_curves2dData.values()) {
        bool ret = plot2dData->append(obj);
        if (ret)
            plot2dData->setUpdatedFlagToTrue();

    }

    foreach(Plot3dData* plot3dData, m_curves3dData.values()) {
        bool ret = plot3dData->append(obj);
        if (ret)
            plot3dData->setUpdatedFlagToTrue();
    }
}



/**
 * @brief ScopeGadgetWidget::replotNewData
 */
void ScopeGadgetWidget::replotNewData()
{
    // If the plot is not visible, do not replot
    if (!isVisible())
        return;

    QMutexLocker locker(&mutex);

    foreach(Plot2dData* plot2dData, m_curves2dData.values())
    {
        if (plot2dData->plotType() == SCATTERPLOT2D){
            plot2dData->removeStaleData();
            switch (m_plot2dType){
            case SCATTERPLOT2D:
            {
                ScatterplotData *scatterplotData = (ScatterplotData*) plot2dData;
                //Plot new data
                if (scatterplotData->readAndResetUpdatedFlag() == true)
                    scatterplotData->curve->setSamples(*(scatterplotData->getXData()), *(scatterplotData->getYData()));

                // Advance axis in case of time series plot
                if (m_Scatterplot2dType == TIMESERIES2D){
                    QDateTime NOW = QDateTime::currentDateTime();
                    double toTime = NOW.toTime_t();
                    toTime += NOW.time().msec() / 1000.0;

                    setAxisScale(QwtPlot::xBottom, toTime - m_xWindowSize, toTime);
                }
                break;
            }
            default:
                //We shouldn't be able to get this far. This means that somewhere the plot types and plot dimensions have gotten out of sync
                Q_ASSERT(0);
            }
        }
        else if (plot2dData->plotType() == HISTOGRAM){
            switch (m_plot2dType){
            case HISTOGRAM:
            {
                //Plot new data
                HistogramData *histogramData = (HistogramData*) plot2dData;
                histogramData->histogram->setData(histogramData->intervalSeriesData);
                histogramData->intervalSeriesData->setSamples(*histogramData->histogramBins); // <-- Is this a memory leak?
                break;
            }
            default:
                //We shouldn't be able to get this far. This means that somewhere the plot types and plot dimensions have gotten out of sync
                Q_ASSERT(0);
            }
        }

    }

    foreach(Plot3dData* plot3dData, m_curves3dData.values())
    {
        if (plot3dData->plotType() == SCATTERPLOT3D){
            plot3dData->removeStaleData();
            if (m_plot3dType == SCATTERPLOT3D){
                //Plot new data
//                plot3dData->curve->setSamples(*plot3dData->xData, *plot3dData->yData);
            }
        }
        if (plot3dData->plotType() == SPECTROGRAM){
            plot3dData->removeStaleData();
            if (m_plot3dType == SPECTROGRAM){
                // Load spectrogram parameters
                SpectrogramData *spectrogramData = (SpectrogramData*) plot3dData;

                // Check for new data
                if (spectrogramData->readAndResetUpdatedFlag() == true){
                    // Plot new data
                    spectrogramData->rasterData->setValueMatrix(*plot3dData->zDataHistory, spectrogramData->windowWidth);

                    // Check autoscale. (For some reason, QwtSpectrogram doesn't support autoscale)
                    if (plot3dData->getZMaximum() == 0){
                        double newVal = spectrogramData->readAndResetAutoscaleValue();
                        if (newVal != 0){
                            spectrogramData->rightAxis->setColorMap( QwtInterval(0, newVal), new ColorMap() );
                            setAxisScale( QwtPlot::yRight, 0, newVal);
                        }
                    }
                }

            }
        }
    }

    replot();
}


/**
 * @brief ScopeGadgetWidget::clearCurvePlots
 */
void ScopeGadgetWidget::clearCurvePlots()
{
    m_grid->detach();

    qDebug() << "length: " << m_curves2dData.size();
    foreach(Plot2dData* plot2dData, m_curves2dData.values()) {

        if (plot2dData->plotType() == SCATTERPLOT2D){
            ScatterplotData *scatterplotData = (ScatterplotData*) plot2dData;
            scatterplotData->curve->detach();

            delete scatterplotData->curve;
            delete scatterplotData;
        }
        else if (plot2dData->plotType() == HISTOGRAM){
            HistogramData *histogramData = (HistogramData*) plot2dData;

            histogramData->histogram->detach();

            // Delete data bins
            delete histogramData->histogramBins;
            delete histogramData->histogramInterval;
            // Don't delete intervalSeriesData, this is done by the histogram's destructor
            /* delete histogramData->intervalSeriesData; */

            // Delete histogram (also deletes intervalSeriesData)
            delete histogramData->histogram;

            delete histogramData;
        }
    }

    foreach(Plot3dData* plot3dData, m_curves3dData.values()) {
        if (plot3dData->plotType() == SCATTERPLOT3D){
            plot3dData->curve->detach();

            delete plot3dData->curve;
            delete plot3dData;
        }
        else if (plot3dData->plotType() == SPECTROGRAM){
            SpectrogramData* spectrogramData = (SpectrogramData*) plot3dData;
            spectrogramData->spectrogram->detach();

            // Don't delete raster data, this is done by the spectrogram's destructor
            /* delete plot3dData->rasterData; */

            // Delete spectrogram (also deletes raster data)
            delete spectrogramData->spectrogram;
            delete spectrogramData;
        }
    }

    m_curves2dData.clear();
    m_curves3dData.clear();
}


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

void ScopeGadgetWidget::showEvent(QShowEvent *event)
{
    replotNewData();
    QwtPlot::showEvent(event);
}
