/**
 ******************************************************************************
 *
 * @file       scatterplotscope.cpp
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

#include "scatterplotdata.h"
#include "scopes2d/scatterplotscopeconfig.h"

#include "uavtalk/telemetrymanager.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include "coreplugin/icore.h"
#include "coreplugin/connectionmanager.h"


Scatterplot2dScope::Scatterplot2dScope()
{
    scatterplot2dType = TIMESERIES2D;
    xAxisUnits = "";
    m_refreshInterval = 50; //TODO: This should not be set here. Probably should come from a define somewhere.
    timeHorizon = 60;
}


Scatterplot2dScope::Scatterplot2dScope(QSettings *qSettings) //TODO: Understand where to put m_refreshInterval default values
{
    this->m_refreshInterval = m_refreshInterval;
    scatterplot2dType =  (Scatterplot2dType) qSettings->value("scatterplot2dType").toUInt();
    timeHorizon = qSettings->value("timeHorizon").toDouble();

    int dataSourceCount = qSettings->value("dataSourceCount").toInt();
    for(int i = 0; i < dataSourceCount; i++)
    {
        qSettings->beginGroup(QString("scatterplotDataSource") + QString().number(i));

        Plot2dCurveConfiguration *plotCurveConf = new Plot2dCurveConfiguration();

        plotCurveConf->uavObjectName = qSettings->value("uavObject").toString();
        plotCurveConf->uavFieldName  = qSettings->value("uavField").toString();
        plotCurveConf->color         = qSettings->value("color").value<QRgb>();
        plotCurveConf->yScalePower   = qSettings->value("yScalePower").toInt();
        plotCurveConf->mathFunction  = qSettings->value("mathFunction").toString();
        plotCurveConf->yMeanSamples  = qSettings->value("yMeanSamples").toInt();
        plotCurveConf->yMinimum      = qSettings->value("yMinimum").toDouble();
        plotCurveConf->yMaximum      = qSettings->value("yMaximum").toDouble();

        //Stop reading XML block
        qSettings->endGroup();

        m_scatterplotSourceConfigs.append(plotCurveConf);

    }
}

Scatterplot2dScope::Scatterplot2dScope(Ui::ScopeGadgetOptionsPage *options_page)
{
    bool parseOK = false;

    timeHorizon = options_page->spnDataSize->value();
    scatterplot2dType = (Scatterplot2dType) options_page->cmbXAxisScatterplot2d->itemData(options_page->cmbXAxisScatterplot2d->currentIndex()).toUInt();

    for(int iIndex = 0; iIndex < options_page->lst2dCurves->count();iIndex++) {
        QListWidgetItem* listItem = options_page->lst2dCurves->item(iIndex);

        Plot2dCurveConfiguration* newPlotCurveConfigs = new Plot2dCurveConfiguration();
        newPlotCurveConfigs->uavObjectName = listItem->data(Qt::UserRole + 0).toString();
        newPlotCurveConfigs->uavFieldName  = listItem->data(Qt::UserRole + 1).toString();
        newPlotCurveConfigs->yScalePower  = listItem->data(Qt::UserRole + 2).toInt(&parseOK);
        if(!parseOK)
            newPlotCurveConfigs->yScalePower = 0;

        QVariant varColor  = listItem->data(Qt::UserRole + 3);
        int rgb = varColor.toInt(&parseOK);
        if(!parseOK)
            newPlotCurveConfigs->color = QColor(Qt::black).rgb();
        else
            newPlotCurveConfigs->color = (QRgb)rgb;

        newPlotCurveConfigs->yMeanSamples = listItem->data(Qt::UserRole + 4).toInt(&parseOK);
        if(!parseOK)
            newPlotCurveConfigs->yMeanSamples = 1;

        newPlotCurveConfigs->mathFunction  = listItem->data(Qt::UserRole + 5).toString();


        m_scatterplotSourceConfigs.append(newPlotCurveConfigs);
    }

}


Scatterplot2dScope::~Scatterplot2dScope()
{

}


ScopesGeneric* Scatterplot2dScope::cloneScope(ScopesGeneric *originalScope)
{
    Scatterplot2dScope *originalScatterplot2dScope = (Scatterplot2dScope*) originalScope;
    Scatterplot2dScope *cloneObj = new Scatterplot2dScope();

    cloneObj->m_refreshInterval = originalScatterplot2dScope->m_refreshInterval;
    cloneObj->timeHorizon = originalScatterplot2dScope->timeHorizon;
    cloneObj->scatterplot2dType = originalScatterplot2dScope->scatterplot2dType;

    int histogramSourceCount = originalScatterplot2dScope->m_scatterplotSourceConfigs.size();

    for(int i = 0; i < histogramSourceCount; i++)
    {
        Plot2dCurveConfiguration *currentScatterplotSourceConf = originalScatterplot2dScope->m_scatterplotSourceConfigs.at(i);
        Plot2dCurveConfiguration *newScatterplotSourceConf     = new Plot2dCurveConfiguration();

        newScatterplotSourceConf->uavObjectName = currentScatterplotSourceConf->uavObjectName;
        newScatterplotSourceConf->uavFieldName  = currentScatterplotSourceConf->uavFieldName;
        newScatterplotSourceConf->color         = currentScatterplotSourceConf->color;
        newScatterplotSourceConf->yScalePower   = currentScatterplotSourceConf->yScalePower;
        newScatterplotSourceConf->yMeanSamples  = currentScatterplotSourceConf->yMeanSamples;
        newScatterplotSourceConf->mathFunction  = currentScatterplotSourceConf->mathFunction;
        newScatterplotSourceConf->yMinimum = currentScatterplotSourceConf->yMinimum;
        newScatterplotSourceConf->yMaximum = currentScatterplotSourceConf->yMaximum;

        cloneObj->m_scatterplotSourceConfigs.append(newScatterplotSourceConf);
    }

    return cloneObj;
}

void Scatterplot2dScope::saveConfiguration(QSettings* qSettings)
{
    //Stop writing XML blocks
    qSettings->beginGroup(QString("plot2d"));
//    qSettings->beginGroup(QString("scatterplot"));

    qSettings->setValue("timeHorizon", timeHorizon);
    qSettings->setValue("plot2dType", SCATTERPLOT2D);
    qSettings->setValue("scatterplot2dType", scatterplot2dType);

    int dataSourceCount = m_scatterplotSourceConfigs.size();
    qSettings->setValue("dataSourceCount", dataSourceCount);

    // For each curve source in the plot
    for(int i = 0; i < dataSourceCount; i++)
    {
        Plot2dCurveConfiguration *plotCurveConf = m_scatterplotSourceConfigs.at(i); //TODO: Understand why this seems to be grabbing i-1
        qSettings->beginGroup(QString("scatterplotDataSource") + QString().number(i));

        qSettings->setValue("uavObject",  plotCurveConf->uavObjectName);
        qSettings->setValue("uavField",  plotCurveConf->uavFieldName);
        qSettings->setValue("color",  plotCurveConf->color);
        qSettings->setValue("mathFunction",  plotCurveConf->mathFunction);
        qSettings->setValue("yScalePower",  plotCurveConf->yScalePower);
        qSettings->setValue("yMeanSamples",  plotCurveConf->yMeanSamples);
        qSettings->setValue("yMinimum",  plotCurveConf->yMinimum);
        qSettings->setValue("yMaximum",  plotCurveConf->yMaximum);

        //Stop writing XML blocks
        qSettings->endGroup();
    }
    //Stop writing XML blocks
//    qSettings->endGroup();
    qSettings->endGroup();
}


/**
 * @brief Scatterplot2dScope::replaceScatterplotDataSource Replaces the list of histogram data sources
 * @param scatterplotSourceConfigs
 */
void Scatterplot2dScope::replaceScatterplotDataSource(QList<Plot2dCurveConfiguration*> scatterplotSourceConfigs)
{
    m_scatterplotSourceConfigs.clear();
    m_scatterplotSourceConfigs.append(scatterplotSourceConfigs);
}


/**
 * @brief Scatterplot2dScope::loadConfiguration loads the plot configuration into the scope gadget widget
 * @param scopeGadgetWidget
 */
void Scatterplot2dScope::loadConfiguration(ScopeGadgetWidget **scopeGadgetWidget)
{
    switch (scatterplot2dType)
    {
    case SERIES2D:
        (*scopeGadgetWidget)->setupSeriesPlot(this);
        break;
    case TIMESERIES2D:
        (*scopeGadgetWidget)->setupTimeSeriesPlot(this);
        break;
    default:
        //We shouldn't be able to get here.
        Q_ASSERT(0);
    }

    (*scopeGadgetWidget)->setRefreshInterval(m_refreshInterval);
    (*scopeGadgetWidget)->setXWindowSize(timeHorizon);

    // Configured each data source
    foreach (Plot2dCurveConfiguration* plotCurveConfig,  m_scatterplotSourceConfigs)
    {
        QString uavObjectName = plotCurveConfig->uavObjectName;
        QString uavFieldName = plotCurveConfig->uavFieldName;
        int scaleOrderFactor = plotCurveConfig->yScalePower;
        int meanSamples = plotCurveConfig->yMeanSamples;
        QString mathFunction = plotCurveConfig->mathFunction;
        QRgb color = plotCurveConfig->color;

        QPen pen(  QBrush(QColor(color),Qt::SolidPattern),
           (qreal)1,
           Qt::SolidLine,
           Qt::SquareCap,
           Qt::BevelJoin);

        // This used to be a separate function called add2dCurvePlot(). It probably still could/should be.
        {
            ScatterplotData* scatterplotData;

            switch(scatterplot2dType){
            case SERIES2D:
                scatterplotData = new SeriesPlotData(uavObjectName, uavFieldName);
                break;
            case TIMESERIES2D:
                scatterplotData = new TimeSeriesPlotData(uavObjectName, uavFieldName);
                break;
            }

            scatterplotData->setXWindowSize((*scopeGadgetWidget)->m_xWindowSize);
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
            QString units = getUavObjectFieldUnits(scatterplotData->getUavoName(), scatterplotData->getUavoFieldName());

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
            plotCurve->attach((*scopeGadgetWidget));
            scatterplotData->curve = plotCurve;

            //Keep the curve details for later
            m_curves2dData.insert(curveNameScaledMath, scatterplotData);

            //Link to the new signal data only if this UAVObject has not been connected yet
            if (!(*scopeGadgetWidget)->m_connectedUAVObjects.contains(obj->getName())) {
                (*scopeGadgetWidget)->m_connectedUAVObjects.append(obj->getName());
                connect(obj, SIGNAL(objectUpdated(UAVObject*)), (*scopeGadgetWidget), SLOT(uavObjectReceived(UAVObject*)));
            }

        }
    }
    mutex.lock();
    (*scopeGadgetWidget)->replot();
    mutex.unlock();
}


void Scatterplot2dScope::setGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page)
{
    //Set the tab widget to 2D
    options_page->tabWidget2d3d->setCurrentWidget(options_page->tabPlot2d);
//    on_tabWidget2d3d_currentIndexChanged(options_page->tabWidget2d3d->currentIndex());

    //Set the plot type
    options_page->cmb2dPlotType->setCurrentIndex(options_page->cmb2dPlotType->findData(getScopeType()));
//    on_cmb2dPlotType_currentIndexChanged(options_page->cmb2dPlotType->currentText());

    //add the configured 2D curves
    options_page->lst2dCurves->clear();  //Clear list first

    foreach (Plot2dCurveConfiguration* plotData, m_scatterplotSourceConfigs) {
        options_page->cmbXAxisScatterplot2d->setCurrentIndex(scatterplot2dType);
        options_page->spnDataSize->setValue(timeHorizon);

        QString uavObjectName = plotData->uavObjectName;
        QString uavFieldName = plotData->uavFieldName;
        int scale = plotData->yScalePower;
        int mean = plotData->yMeanSamples;
        QString mathFunction = plotData->mathFunction;
        QVariant varColor = plotData->color;

        //TODO: Refer this back to scopegadgetoptionspage.
//        addPlot2dCurveConfig(uavObjectName,uavFieldName,scale,mean,mathFunction,varColor);
        {
            QString listItemDisplayText = uavObjectName + "." + uavFieldName; // Generate the name
            options_page->lst2dCurves->addItem(listItemDisplayText);  // Add the name to the list
            int itemIdx = options_page->lst2dCurves->count() - 1; // Get the index number for the new value
            QListWidgetItem *listWidgetItem = options_page->lst2dCurves->item(itemIdx); //Find the widget item

            //Apply all settings to curve
//            setPlot2dCurveProperties(listWidgetItem, uavObjectName, uavFieldName, scale, mean, mathFunction, varColor);
            {
                bool parseOK = false;
                QString listItemDisplayText;
                QRgb rgbColor;

                if(uavObjectName!="")
                {
                    //Set the properties of the newly added list item
                    listItemDisplayText = uavObjectName + "." + uavFieldName;
                    rgbColor = (QRgb)varColor.toInt(&parseOK);
                    if(!parseOK)
                        rgbColor = qRgb(255,0,0);
                }
                else{
                    listItemDisplayText = "New graph";
                    rgbColor = qRgb(255,0,0);
                }

                QColor color = QColor( rgbColor );
                listWidgetItem->setText(listItemDisplayText);
                listWidgetItem->setTextColor( color );

                //Store some additional data for the plot curve on the list item
                listWidgetItem->setData(Qt::UserRole + 0,QVariant(uavObjectName));
                listWidgetItem->setData(Qt::UserRole + 1,QVariant(uavFieldName));
                listWidgetItem->setData(Qt::UserRole + 2,QVariant(scale));
                listWidgetItem->setData(Qt::UserRole + 3,varColor);
                listWidgetItem->setData(Qt::UserRole + 4,QVariant(mean));
                listWidgetItem->setData(Qt::UserRole + 5,QVariant(mathFunction));
            }

            //Select the row with the new name
            options_page->lst2dCurves->setCurrentRow(itemIdx);
        }
    }

    //Select row 1st row in list
    options_page->lst2dCurves->setCurrentRow(0, QItemSelectionModel::ClearAndSelect);

}



void Scatterplot2dScope::preparePlot(ScopeGadgetWidget *scopeGadgetWidget)
{
    scopeGadgetWidget->setMinimumSize(64, 64);
    scopeGadgetWidget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    scopeGadgetWidget->setCanvasBackground(QColor(64, 64, 64));



    //Add grid lines
    scopeGadgetWidget->m_grid->enableX( true );
    scopeGadgetWidget->m_grid->enableY( true );
    scopeGadgetWidget->m_grid->enableXMin( false );
    scopeGadgetWidget->m_grid->enableYMin( false );
    scopeGadgetWidget->m_grid->setMajPen(QPen(Qt::gray, 0, Qt::DashLine));
    scopeGadgetWidget->m_grid->setMinPen(QPen(Qt::lightGray, 0, Qt::DotLine));
    scopeGadgetWidget->m_grid->setPen(QPen(Qt::darkGray, 1, Qt::DotLine));
    scopeGadgetWidget->m_grid->attach(scopeGadgetWidget);

    // Add the legend
    scopeGadgetWidget->addLegend();

    // Only start the timer if we are already connected
    Core::ConnectionManager *cm = Core::ICore::instance()->connectionManager();
    if (cm->getCurrentConnection() && scopeGadgetWidget->replotTimer)
    {
        if (!scopeGadgetWidget->replotTimer->isActive())
            scopeGadgetWidget->replotTimer->start(m_refreshInterval);
        else
            scopeGadgetWidget->replotTimer->setInterval(m_refreshInterval);
    }
}

void Scatterplot2dScope::plotNewData(ScopeGadgetWidget *scopeGadgetWidget)
{

    bool updateXAxisFlag = true;

    foreach(Plot2dData* plot2dData, m_curves2dData.values())
    {
        ScatterplotData *scatterplotData = (ScatterplotData*) plot2dData;
        //Plot new data
        if (scatterplotData->readAndResetUpdatedFlag() == true)
            scatterplotData->curve->setSamples(*(scatterplotData->getXData()), *(scatterplotData->getYData()));

        // Advance axis in case of time series plot. // TODO: Do this just once.
        if (scatterplot2dType == TIMESERIES2D && updateXAxisFlag == true)
        {
            QDateTime NOW = QDateTime::currentDateTime();
            double toTime = NOW.toTime_t();
            toTime += NOW.time().msec() / 1000.0;

            scopeGadgetWidget->setAxisScale(QwtPlot::xBottom, toTime - scopeGadgetWidget->m_xWindowSize, toTime);
            updateXAxisFlag = false;
        }
//        else if (plot2dData->plotType() == HISTOGRAM)
//        {
//            switch (m_plot2dType){
//            case HISTOGRAMs:
//            {
//                //Plot new data
//                HistogramData *histogramData = (HistogramData*) plot2dData;
//                histogramData->histogram->setData(histogramData->intervalSeriesData);
//                histogramData->intervalSeriesData->setSamples(*histogramData->histogramBins); // <-- Is this a memory leak?
//                break;
//            }
//        }

    }

}


void Scatterplot2dScope::clearPlots()
{
    foreach(Plot2dData* plot2dData, m_curves2dData.values()) {
        ScatterplotData *scatterplotData = (ScatterplotData*) plot2dData;
        scatterplotData->curve->detach();

        delete scatterplotData->curve;
        delete scatterplotData;
    }

    // Clear the data
    m_curves2dData.clear();
}



void Scatterplot2dScope::uavObjectReceived(UAVObject* obj)
{
    foreach(Plot2dData* plot2dData, m_curves2dData.values()) {
        bool ret = plot2dData->append(obj);
        if (ret)
            plot2dData->setUpdatedFlagToTrue();
    }
}
