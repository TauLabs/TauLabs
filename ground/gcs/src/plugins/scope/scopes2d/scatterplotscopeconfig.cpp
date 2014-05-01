/**
 ******************************************************************************
 *
 * @file       scatterplotscope.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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
#include "scopegadgetoptionspage.h"

#include "uavtalk/telemetrymanager.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include "coreplugin/icore.h"
#include "coreplugin/connectionmanager.h"


/**
 * @brief Scatterplot2dScopeConfig::Scatterplot2dScopeConfig Default constructor.
 */
Scatterplot2dScopeConfig::Scatterplot2dScopeConfig()
{
    scatterplot2dType = TIMESERIES2D;
    m_refreshInterval = 50;
    timeHorizon = 60;
}


/**
 * @brief Scatterplot2dScopeConfig::Scatterplot2dScopeConfig Constructor using the XML settings
 * @param qSettings settings XML object
 */
Scatterplot2dScopeConfig::Scatterplot2dScopeConfig(QSettings *qSettings)
{
    this->m_refreshInterval = m_refreshInterval;
    scatterplot2dType =  (Scatterplot2dType) qSettings->value("scatterplot2dType").toUInt();
    timeHorizon = qSettings->value("timeHorizon").toDouble();

    int dataSourceCount = qSettings->value("dataSourceCount").toInt();
    for(int i = 0; i < dataSourceCount; i++)
    {
        // Start reading XML block
        qSettings->beginGroup(QString("scatterplotDataSource") + QString().number(i));

        Plot2dCurveConfiguration *plotCurveConf = new Plot2dCurveConfiguration();

        plotCurveConf->uavObjectName = qSettings->value("uavObject").toString();
        plotCurveConf->uavFieldName  = qSettings->value("uavField").toString();
        plotCurveConf->color         = qSettings->value("color").value<QRgb>();
        plotCurveConf->yScalePower   = qSettings->value("yScalePower").toInt();
        plotCurveConf->mathFunction  = qSettings->value("mathFunction").toString();
        plotCurveConf->yMeanSamples  = qSettings->value("yMeanSamples").toUInt();

        // Stop reading XML block
        qSettings->endGroup();

        m_scatterplotSourceConfigs.append(plotCurveConf);
    }
}


/**
 * @brief Scatterplot2dScopeConfig::Scatterplot2dScopeConfig Constructor using the GUI settings
 * @param options_page GUI settings preference pane
 */
Scatterplot2dScopeConfig::Scatterplot2dScopeConfig(Ui::ScopeGadgetOptionsPage *options_page)
{
    bool parseOK = false;

    timeHorizon = options_page->spnDataSize->value();
    scatterplot2dType = (Scatterplot2dType) options_page->cmbXAxisScatterplot2d->itemData(options_page->cmbXAxisScatterplot2d->currentIndex()).toInt();

    for(int iIndex = 0; iIndex < options_page->lst2dCurves->count();iIndex++) {
        QListWidgetItem* listItem = options_page->lst2dCurves->item(iIndex);

        //Store some additional data for the plot curve on the list item
        Plot2dCurveConfiguration* newPlotCurveConfigs = new Plot2dCurveConfiguration();
        newPlotCurveConfigs->uavObjectName = listItem->data(Qt::UserRole + ScopeGadgetOptionsPage::UR_UAVOBJECT).toString();
        newPlotCurveConfigs->uavFieldName  = listItem->data(Qt::UserRole + ScopeGadgetOptionsPage::UR_UAVFIELD).toString();
        newPlotCurveConfigs->yScalePower  = listItem->data(Qt::UserRole + ScopeGadgetOptionsPage::UR_SCALE).toInt(&parseOK);
        if(!parseOK)
            newPlotCurveConfigs->yScalePower = 0;

        QVariant varColor  = listItem->data(Qt::UserRole + ScopeGadgetOptionsPage::UR_COLOR);
        int rgb = varColor.toInt(&parseOK);
        if(!parseOK)
            newPlotCurveConfigs->color = QColor(Qt::black).rgb();
        else
            newPlotCurveConfigs->color = (QRgb)rgb;

        newPlotCurveConfigs->yMeanSamples = listItem->data(Qt::UserRole + ScopeGadgetOptionsPage::UR_MEAN).toUInt(&parseOK);
        if(!parseOK)
            newPlotCurveConfigs->yMeanSamples = 1;

        newPlotCurveConfigs->mathFunction  = listItem->data(Qt::UserRole + ScopeGadgetOptionsPage::UR_MATHFUNCTION).toString();


        m_scatterplotSourceConfigs.append(newPlotCurveConfigs);
    }

}


Scatterplot2dScopeConfig::~Scatterplot2dScopeConfig()
{

}


/**
 * @brief Scatterplot2dScopeConfig::cloneScope Clones scope from existing GUI configuration
 * @param originalScope
 * @return
 */
ScopeConfig* Scatterplot2dScopeConfig::cloneScope(ScopeConfig *originalScope)
{
    Scatterplot2dScopeConfig *originalScatterplot2dScopeConfig = (Scatterplot2dScopeConfig*) originalScope;
    Scatterplot2dScopeConfig *cloneObj = new Scatterplot2dScopeConfig();

    cloneObj->m_refreshInterval = originalScatterplot2dScopeConfig->m_refreshInterval;
    cloneObj->timeHorizon = originalScatterplot2dScopeConfig->timeHorizon;
    cloneObj->scatterplot2dType = originalScatterplot2dScopeConfig->scatterplot2dType;

    int scatterplotSourceCount = originalScatterplot2dScopeConfig->m_scatterplotSourceConfigs.size();

    for(int i = 0; i < scatterplotSourceCount; i++)
    {
        Plot2dCurveConfiguration *currentScatterplotSourceConf = originalScatterplot2dScopeConfig->m_scatterplotSourceConfigs.at(i);
        Plot2dCurveConfiguration *newScatterplotSourceConf     = new Plot2dCurveConfiguration();

        newScatterplotSourceConf->uavObjectName = currentScatterplotSourceConf->uavObjectName;
        newScatterplotSourceConf->uavFieldName  = currentScatterplotSourceConf->uavFieldName;
        newScatterplotSourceConf->color         = currentScatterplotSourceConf->color;
        newScatterplotSourceConf->yScalePower   = currentScatterplotSourceConf->yScalePower;
        newScatterplotSourceConf->yMeanSamples  = currentScatterplotSourceConf->yMeanSamples;
        newScatterplotSourceConf->mathFunction  = currentScatterplotSourceConf->mathFunction;

        cloneObj->m_scatterplotSourceConfigs.append(newScatterplotSourceConf);
    }

    return cloneObj;
}


/**
 * @brief Scatterplot2dScopeConfig::saveConfiguration Saves configuration to XML file
 * @param qSettings
 */
void Scatterplot2dScopeConfig::saveConfiguration(QSettings* qSettings)
{
    //Stop writing XML blocks
    qSettings->beginGroup(QString("plot2d"));

    qSettings->setValue("timeHorizon", timeHorizon);
    qSettings->setValue("plot2dType", SCATTERPLOT2D);
    qSettings->setValue("scatterplot2dType", scatterplot2dType);

    int dataSourceCount = m_scatterplotSourceConfigs.size();
    qSettings->setValue("dataSourceCount", dataSourceCount);

    // For each curve source in the plot
    for(int i = 0; i < dataSourceCount; i++)
    {
        Plot2dCurveConfiguration *plotCurveConf = m_scatterplotSourceConfigs.at(i);
        qSettings->beginGroup(QString("scatterplotDataSource") + QString().number(i));

        qSettings->setValue("uavObject",  plotCurveConf->uavObjectName);
        qSettings->setValue("uavField",  plotCurveConf->uavFieldName);
        qSettings->setValue("color",  plotCurveConf->color);
        qSettings->setValue("mathFunction",  plotCurveConf->mathFunction);
        qSettings->setValue("yScalePower",  plotCurveConf->yScalePower);
        qSettings->setValue("yMeanSamples",  plotCurveConf->yMeanSamples);

        //Stop writing XML blocks
        qSettings->endGroup();
    }
    //Stop writing XML blocks
    qSettings->endGroup();
}


/**
 * @brief Scatterplot2dScopeConfig::replaceScatterplotDataSource Replaces the list of scatterplot data sources
 * @param scatterplotSourceConfigs
 */
void Scatterplot2dScopeConfig::replaceScatterplotDataSource(QList<Plot2dCurveConfiguration*> scatterplotSourceConfigs)
{
    m_scatterplotSourceConfigs.clear();
    m_scatterplotSourceConfigs.append(scatterplotSourceConfigs);
}


/**
 * @brief Scatterplot2dScopeConfig::loadConfiguration loads the plot configuration into the scope gadget widget
 * @param scopeGadgetWidget
 */
void Scatterplot2dScopeConfig::loadConfiguration(ScopeGadgetWidget *scopeGadgetWidget)
{
    preparePlot(scopeGadgetWidget);
    scopeGadgetWidget->setScope(this);
    scopeGadgetWidget->startTimer(m_refreshInterval);


    // Configure each data source
    foreach (Plot2dCurveConfiguration* plotCurveConfig,  m_scatterplotSourceConfigs)
    {
        QString uavObjectName = plotCurveConfig->uavObjectName;
        QString uavFieldName = plotCurveConfig->uavFieldName;
        QRgb color = plotCurveConfig->color;

        ScatterplotData* scatterplotData = NULL;

        switch(scatterplot2dType){
        case SERIES2D:
            scatterplotData = new SeriesPlotData(uavObjectName, uavFieldName);
            break;
        case TIMESERIES2D:
            scatterplotData = new TimeSeriesPlotData(uavObjectName, uavFieldName);
            break;
        }

        scatterplotData->setXWindowSize(timeHorizon);
        scatterplotData->setScalePower(plotCurveConfig->yScalePower);
        scatterplotData->setMeanSamples(plotCurveConfig->yMeanSamples);
        scatterplotData->setMathFunction(plotCurveConfig->mathFunction);

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
        if(plotCurveConfig->yScalePower == 0)
            curveNameScaled = curveName + "(" + units + ")";
        else
            curveNameScaled = curveName + "(x10^" + QString::number(plotCurveConfig->yScalePower) + " " + units + ")";

        QString curveNameScaledMath;
        if (plotCurveConfig->mathFunction == "None")
            curveNameScaledMath = curveNameScaled;
        else if (plotCurveConfig->mathFunction == "Boxcar average"){
            curveNameScaledMath = curveNameScaled + " (avg)";
        }
        else if (plotCurveConfig->mathFunction == "Standard deviation"){
            curveNameScaledMath = curveNameScaled + " (std)";
        }
        else
        {
            //Shouldn't be able to get here. Perhaps a new math function was added without
            // updating this list?
            Q_ASSERT(0);
        }

        while(scopeGadgetWidget->getDataSources().keys().contains(curveNameScaledMath))
            curveNameScaledMath=curveNameScaledMath+"*";

        //Create the curve plot
        QwtPlotCurve* plotCurve = new QwtPlotCurve(curveNameScaledMath);
        plotCurve->setPen(QPen(QBrush(QColor(color), Qt::SolidPattern), (qreal)1, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
        plotCurve->setSamples(*(scatterplotData->getXData()), *(scatterplotData->getYData()));
        plotCurve->attach(scopeGadgetWidget);
        scatterplotData->setCurve(plotCurve);

        //Keep the curve details for later
        scopeGadgetWidget->insertDataSources(curveNameScaledMath, scatterplotData);

        // Connect the UAVO
        scopeGadgetWidget->connectUAVO(obj);
    }
    mutex.lock();
    scopeGadgetWidget->replot();
    mutex.unlock();
}



/**
 * @brief Scatterplot2dScopeConfig::setGuiConfigurationSet the GUI elements based on values from the XML settings file
 * @param options_page
 */
void Scatterplot2dScopeConfig::setGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page)
{
    //Set the tab widget to 2D
    options_page->tabWidget2d3d->setCurrentWidget(options_page->tabPlot2d);

    //Set the plot type
    options_page->cmb2dPlotType->setCurrentIndex(options_page->cmb2dPlotType->findData(SCATTERPLOT2D));

    //add the configured 2D curves
    options_page->lst2dCurves->clear();  //Clear list first

    foreach (Plot2dCurveConfiguration* plotData, m_scatterplotSourceConfigs) {
        options_page->cmbXAxisScatterplot2d->setCurrentIndex(scatterplot2dType);
        options_page->spnDataSize->setValue(timeHorizon);

        QString uavObjectName = plotData->uavObjectName;
        QString uavFieldName = plotData->uavFieldName;
        int scale = plotData->yScalePower;
        unsigned int mean = plotData->yMeanSamples;
        QString mathFunction = plotData->mathFunction;
        QVariant varColor = plotData->color;

        QString listItemDisplayText = uavObjectName + "." + uavFieldName; // Generate the name
        options_page->lst2dCurves->addItem(listItemDisplayText);  // Add the name to the list
        int itemIdx = options_page->lst2dCurves->count() - 1; // Get the index number for the new value
        QListWidgetItem *listWidgetItem = options_page->lst2dCurves->item(itemIdx); //Find the widget item

        bool parseOK = false;
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
        listWidgetItem->setData(Qt::UserRole + ScopeGadgetOptionsPage::UR_UAVOBJECT, QVariant(uavObjectName));
        listWidgetItem->setData(Qt::UserRole + ScopeGadgetOptionsPage::UR_UAVFIELD, QVariant(uavFieldName));
        listWidgetItem->setData(Qt::UserRole + ScopeGadgetOptionsPage::UR_SCALE, QVariant(scale));
        listWidgetItem->setData(Qt::UserRole + ScopeGadgetOptionsPage::UR_COLOR, varColor);
        listWidgetItem->setData(Qt::UserRole + ScopeGadgetOptionsPage::UR_MEAN, QVariant(mean));
        listWidgetItem->setData(Qt::UserRole + ScopeGadgetOptionsPage::UR_MATHFUNCTION, QVariant(mathFunction));

        //Select the row with the new name
        options_page->lst2dCurves->setCurrentRow(itemIdx);
    }

    //Select row 1st row in list
    options_page->lst2dCurves->setCurrentRow(0, QItemSelectionModel::ClearAndSelect);

}


/**
 * @brief Scatterplot2dScopeConfig::preparePlot Prepares the Qwt plot colors and axes
 * @param scopeGadgetWidget
 */
void Scatterplot2dScopeConfig::preparePlot(ScopeGadgetWidget *scopeGadgetWidget)
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

    // Configure axes
    configureAxes(scopeGadgetWidget);
}


/**
 * @brief Scatterplot2dScopeConfig::configureAxes Configure the axes
 * @param scopeGadgetWidget
 */
void Scatterplot2dScopeConfig::configureAxes(ScopeGadgetWidget *scopeGadgetWidget)
{
    switch (scatterplot2dType)
    {
    case TIMESERIES2D: {
        // Configure axes
        scopeGadgetWidget->setAxisScaleDraw(QwtPlot::xBottom, new TimeScaleDraw());
        uint NOW = QDateTime::currentDateTime().toTime_t();
        scopeGadgetWidget->setAxisScale(QwtPlot::xBottom, NOW - timeHorizon / 1000, NOW);
        break;
    }
    case SERIES2D:
    default:
        scopeGadgetWidget->setAxisScaleDraw(QwtPlot::xBottom, new QwtScaleDraw());
        scopeGadgetWidget->setAxisScale(QwtPlot::xBottom, 0, timeHorizon);
        break;
    }

    scopeGadgetWidget->setAxisAutoScale(QwtPlot::yLeft, true);
    scopeGadgetWidget->setAxisLabelRotation(QwtPlot::xBottom, 0.0);
    scopeGadgetWidget->setAxisLabelAlignment(QwtPlot::xBottom, Qt::AlignLeft | Qt::AlignBottom);
    scopeGadgetWidget->axisWidget( QwtPlot::yRight )->setColorBarEnabled( false );
    scopeGadgetWidget->enableAxis( QwtPlot::yRight, false );



    // Reduce the gap between the scope canvas and the axis scale
    QwtScaleWidget *scaleWidget = scopeGadgetWidget->axisWidget(QwtPlot::xBottom);
    scaleWidget->setMargin(0);

    // Reduce the axis font size
    QFont fnt(scopeGadgetWidget->axisFont(QwtPlot::xBottom));
    fnt.setPointSize(7);
    scopeGadgetWidget->setAxisFont(QwtPlot::xBottom, fnt);	// x-axis
    scopeGadgetWidget->setAxisFont(QwtPlot::yLeft, fnt);	// y-axis
    scopeGadgetWidget->setAxisFont(QwtPlot::yRight, fnt);	// y-axis

    /*
     In situations, when there is a label at the most right position of the
     scale, additional space is needed to display the overlapping part
     of the label would be taken by reducing the width of scale and canvas.
     To avoid this "jumping canvas" effect, we add a permanent margin.
     We don't need to do the same for the left border, because there
     is enough space for the overlapping label below the left scale.
     */
    /*
    const int fmh = QFontMetrics(scaleWidget->font()).height();
    scaleWidget->setMinBorderDist(0, fmh / 2);

    const int fmw = QFontMetrics(scaleWidget->font()).width(" 00:00:00 ");
    const int fmw = QFontMetrics(scaleWidget->font()).width(" ");
    scaleWidget->setMinBorderDist(0, fmw);
    */
}
