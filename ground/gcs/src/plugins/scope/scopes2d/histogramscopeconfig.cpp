/**
 ******************************************************************************
 *
 * @file       histogramscopeconfig.cpp
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

#include "histogramplotdata.h"
#include "scopes2d/histogramscopeconfig.h"
#include "scopegadgetoptionspage.h"

#include "coreplugin/icore.h"
#include "coreplugin/connectionmanager.h"


/**
 * @brief HistogramScopeConfig::HistogramScopeConfig Default constructor
 */
HistogramScopeConfig::HistogramScopeConfig()
{
    binWidth = 1;
    maxNumberOfBins = 1000;
    m_refreshInterval = 50;
}

/**
 * @brief HistogramScopeConfig::HistogramScopeConfig Constructor using the XML settings
 * @param qSettings settings XML object
 */
HistogramScopeConfig::HistogramScopeConfig(QSettings *qSettings)
{
    binWidth    = qSettings->value("binWidth").toDouble();
    //Ensure binWidth is not too small
    if (binWidth < 1e-3)
        binWidth = 1e-3;

    maxNumberOfBins = qSettings->value("maxNumberOfBins").toInt();
    this->m_refreshInterval = m_refreshInterval;
    this->m_plotDimensions = m_plotDimensions;


    int dataSourceCount = qSettings->value("dataSourceCount").toInt();
    for(int i = 0; i < dataSourceCount; i++)
    {
        // Start reading XML block
        qSettings->beginGroup(QString("histogramDataSource") + QString().number(i));

        Plot2dCurveConfiguration *plotCurveConf = new Plot2dCurveConfiguration();

        plotCurveConf->uavObjectName = qSettings->value("uavObject").toString();
        plotCurveConf->uavFieldName  = qSettings->value("uavField").toString();
        plotCurveConf->color         = qSettings->value("color").value<QRgb>();
        plotCurveConf->yScalePower   = qSettings->value("yScalePower").toInt();
        plotCurveConf->mathFunction  = qSettings->value("mathFunction").toString();
        plotCurveConf->yMeanSamples  = qSettings->value("yMeanSamples").toUInt();

        //Stop reading XML block
        qSettings->endGroup();

        m_HistogramSourceConfigs.append(plotCurveConf);

    }
}


/**
 * @brief HistogramScopeConfig::HistogramScopeConfig Constructor using the GUI settings
 * @param options_page GUI settings preference pane
 */
HistogramScopeConfig::HistogramScopeConfig(Ui::ScopeGadgetOptionsPage *options_page)
{
    bool parseOK = false;

    binWidth = options_page->spnBinWidth->value();
    maxNumberOfBins = options_page->spnMaxNumBins->value();

    //For each y-data source in the list
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


        m_HistogramSourceConfigs.append(newPlotCurveConfigs);
    }
}


HistogramScopeConfig::~HistogramScopeConfig()
{

}


/**
 * @brief HistogramScopeConfig::cloneScope Clones scope from existing GUI configuration
 * @param originalScope
 * @return
 */
ScopeConfig* HistogramScopeConfig::cloneScope(ScopeConfig *originalScope)
{
    HistogramScopeConfig *originalHistogramScopeConfig = (HistogramScopeConfig*) originalScope;
    HistogramScopeConfig *cloneObj = new HistogramScopeConfig();

    cloneObj->binWidth = originalHistogramScopeConfig->binWidth;
    cloneObj->maxNumberOfBins = originalHistogramScopeConfig->maxNumberOfBins;
    cloneObj->m_refreshInterval = originalHistogramScopeConfig->m_refreshInterval;

    int histogramSourceCount = originalHistogramScopeConfig->m_HistogramSourceConfigs.size();

    for(int i = 0; i < histogramSourceCount; i++)
    {
        Plot2dCurveConfiguration *currentHistogramSourceConf = originalHistogramScopeConfig->m_HistogramSourceConfigs.at(i);
        Plot2dCurveConfiguration *newHistogramSourceConf     = new Plot2dCurveConfiguration();

        newHistogramSourceConf->uavObjectName = currentHistogramSourceConf->uavObjectName;
        newHistogramSourceConf->uavFieldName  = currentHistogramSourceConf->uavFieldName;
        newHistogramSourceConf->color         = currentHistogramSourceConf->color;
        newHistogramSourceConf->yScalePower   = currentHistogramSourceConf->yScalePower;
        newHistogramSourceConf->yMeanSamples  = currentHistogramSourceConf->yMeanSamples;
        newHistogramSourceConf->mathFunction  = currentHistogramSourceConf->mathFunction;

        cloneObj->m_HistogramSourceConfigs.append(newHistogramSourceConf);
    }

    return cloneObj;
}


/**
 * @brief HistogramScopeConfig::saveConfiguration Saves configuration to XML file
 * @param qSettings
 */
void HistogramScopeConfig::saveConfiguration(QSettings* qSettings)
{
    //Stop writing XML blocks
    qSettings->beginGroup(QString("plot2d"));

    qSettings->setValue("plot2dType", HISTOGRAM);
    qSettings->setValue("binWidth", binWidth);
    qSettings->setValue("maxNumberOfBins", maxNumberOfBins);

    int dataSourceCount = m_HistogramSourceConfigs.size();
    qSettings->setValue("dataSourceCount", dataSourceCount);

    // For each curve source in the plot
    for(int i = 0; i < dataSourceCount; i++)
    {
        Plot2dCurveConfiguration *plotCurveConf = m_HistogramSourceConfigs.at(i);
        qSettings->beginGroup(QString("histogramDataSource") + QString().number(i));

        qSettings->setValue("uavObject",  plotCurveConf->uavObjectName);
        qSettings->setValue("uavField",  plotCurveConf->uavFieldName);
        qSettings->setValue("color",  plotCurveConf->color);
        qSettings->setValue("mathFunction",  plotCurveConf->mathFunction);
        qSettings->setValue("yScalePower",  plotCurveConf->yScalePower);
        qSettings->setValue("yMeanSamples",  plotCurveConf->yMeanSamples);

        //Stop writing XML blocks
        qSettings->endGroup();
    }

    //Stop writing XML block
    qSettings->endGroup();

}


/**
 * @brief HistogramScopeConfig::replaceHistogramSource Replaces the list of histogram data sources
 * @param histogramSourceConfigs
 */
void HistogramScopeConfig::replaceHistogramDataSource(QList<Plot2dCurveConfiguration*> histogramSourceConfigs)
{
    m_HistogramSourceConfigs.clear();
    m_HistogramSourceConfigs.append(histogramSourceConfigs);
}


/**
 * @brief HistogramScopeConfig::loadConfiguration loads the plot configuration into the scope gadget widget
 * @param scopeGadgetWidget
 */
void HistogramScopeConfig::loadConfiguration(ScopeGadgetWidget *scopeGadgetWidget)
{
    preparePlot(scopeGadgetWidget);
    scopeGadgetWidget->setScope(this);
    scopeGadgetWidget->startTimer(m_refreshInterval);

    // Configure each data source
    foreach (Plot2dCurveConfiguration* histogramDataSourceConfig,  m_HistogramSourceConfigs)
    {
        QRgb color = histogramDataSourceConfig->color;

        // Get and store the units
        units = getUavObjectFieldUnits(histogramDataSourceConfig->uavObjectName, histogramDataSourceConfig->uavFieldName);

        HistogramData* histogramData;
        histogramData = new HistogramData(histogramDataSourceConfig->uavObjectName, histogramDataSourceConfig->uavFieldName, binWidth, maxNumberOfBins);

        histogramData->setScalePower(histogramDataSourceConfig->yScalePower);
        histogramData->setMeanSamples(histogramDataSourceConfig->yMeanSamples);
        histogramData->setMathFunction(histogramDataSourceConfig->mathFunction);

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
        QString units = getUavObjectFieldUnits(histogramData->getUavoName(), histogramData->getUavoFieldName());

        //Generate name with scaling factor appeneded
        QString histogramNameScaled;
        if(histogramDataSourceConfig->yScalePower == 0)
            histogramNameScaled = curveName + "(" + units + ")";
        else
            histogramNameScaled = curveName + "(x10^" + QString::number(histogramDataSourceConfig->yScalePower) + " " + units + ")";

        while(scopeGadgetWidget->getDataSources().keys().contains(histogramNameScaled))
            histogramNameScaled=histogramNameScaled+"*";

        // Create the histogram
        QwtPlotHistogram* plotHistogram = new QwtPlotHistogram(histogramNameScaled);
        plotHistogram->setStyle( QwtPlotHistogram::Columns );
        plotHistogram->setBrush(QBrush(QColor(color)));
        plotHistogram->setData( histogramData->getIntervalSeriesData());

        plotHistogram->attach(scopeGadgetWidget);
        histogramData->setHistogram(plotHistogram);

        // Keep the curve details for later
        scopeGadgetWidget->insertDataSources(histogramNameScaled, histogramData);

        // Connect the UAVO
        scopeGadgetWidget->connectUAVO(obj);
    }
    mutex.lock();
    scopeGadgetWidget->replot();
    mutex.unlock();
}


/**
 * @brief HistogramScopeConfig::setGuiConfiguration Set the GUI elements based on values from the XML settings file
 * @param options_page
 */
void HistogramScopeConfig::setGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page)
{
    //Set the tab widget to 2D
    options_page->tabWidget2d3d->setCurrentWidget(options_page->tabPlot2d);

    //Set the plot type
    options_page->cmb2dPlotType->setCurrentIndex(options_page->cmb2dPlotType->findData(HISTOGRAM));

    //add the configured 2D curves
    options_page->lst2dCurves->clear();  //Clear list first

    foreach (Plot2dCurveConfiguration* dataSource,  m_HistogramSourceConfigs) {
        options_page->spnMaxNumBins->setValue(maxNumberOfBins);
        options_page->spnBinWidth->setValue(binWidth);

        QString uavObjectName = dataSource->uavObjectName;
        QString uavFieldName = dataSource->uavFieldName;
        int scale = dataSource->yScalePower;
        unsigned int mean = dataSource->yMeanSamples;
        QString mathFunction = dataSource->mathFunction;
        QVariant varColor = dataSource->color;

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
 * @brief HistogramScopeConfig::preparePlot Prepares the Qwt plot colors and axes
 * @param scopeGadgetWidget
 */
void HistogramScopeConfig::preparePlot(ScopeGadgetWidget *scopeGadgetWidget)
{
    scopeGadgetWidget->setMinimumSize(64, 64);
    scopeGadgetWidget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    scopeGadgetWidget->setCanvasBackground(QColor(64, 64, 64));

    scopeGadgetWidget->plotLayout()->setAlignCanvasToScales( false );

    scopeGadgetWidget->m_grid->enableX( false );
    scopeGadgetWidget->m_grid->enableY( true );
    scopeGadgetWidget->m_grid->enableXMin( false );
    scopeGadgetWidget->m_grid->enableYMin( false );
    scopeGadgetWidget->m_grid->setMajPen( QPen( Qt::black, 0, Qt::DotLine ) );
    scopeGadgetWidget->m_grid->setMinPen(QPen(Qt::lightGray, 0, Qt::DotLine));
    scopeGadgetWidget->m_grid->setPen(QPen(Qt::darkGray, 1, Qt::DotLine));
    scopeGadgetWidget->m_grid->attach( scopeGadgetWidget );

    // Add the legend
    scopeGadgetWidget->addLegend();

    // Configure axes
    configureAxes(scopeGadgetWidget);
}


/**
 * @brief HistogramScopeConfig::configureAxes Configure the axes
 * @param scopeGadgetWidget
 */
void HistogramScopeConfig::configureAxes(ScopeGadgetWidget *scopeGadgetWidget)
{
    // Configure axes
    scopeGadgetWidget->setAxisScaleDraw(QwtPlot::xBottom, new QwtScaleDraw());
    scopeGadgetWidget->setAxisAutoScale(QwtPlot::xBottom);
    scopeGadgetWidget->setAxisLabelRotation(QwtPlot::xBottom, 0.0);
    scopeGadgetWidget->setAxisLabelAlignment(QwtPlot::xBottom, Qt::AlignLeft | Qt::AlignBottom);
    scopeGadgetWidget->axisWidget( QwtPlot::yRight )->setColorBarEnabled( false );
    scopeGadgetWidget->enableAxis( QwtPlot::yRight, false );

    // Reduce the gap between the scope canvas and the axis scale
    QwtScaleWidget *scaleWidget = scopeGadgetWidget->axisWidget(QwtPlot::xBottom);
    scaleWidget->setMargin(0);

    // reduce the axis font size
    QFont fnt(scopeGadgetWidget->axisFont(QwtPlot::xBottom));
    fnt.setPointSize(7);
    scopeGadgetWidget->setAxisFont(QwtPlot::xBottom, fnt);	// x-axis
    scopeGadgetWidget->setAxisFont(QwtPlot::yLeft, fnt);	// y-axis
}
