/**
 ******************************************************************************
 *
 * @file       histogramscope.cpp
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

#include "scopes2d/histogramscopeconfig.h"


HistogramScope::HistogramScope()
{
    binWidth = 1;
    maxNumberOfBins = 1000;
    m_refreshInterval = 50; //TODO: This should not be set here. Probably should come from a define somewhere.
}


HistogramScope::HistogramScope(QSettings *qSettings) //TODO: Understand where to put m_refreshInterval default values
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
        qSettings->beginGroup(QString("histogramDataSource") + QString().number(i));

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

        m_HistogramSourceConfigs.append(plotCurveConf);

    }
}


HistogramScope::HistogramScope(Ui::ScopeGadgetOptionsPage *options_page)
{
    bool parseOK = false;

    binWidth = options_page->spnBinWidth->value();
    maxNumberOfBins = options_page->spnMaxNumBins->value();

    //For each y-data source in the list
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


        m_HistogramSourceConfigs.append(newPlotCurveConfigs);
    }
}

HistogramScope::~HistogramScope()
{

}


ScopesGeneric* HistogramScope::cloneScope(ScopesGeneric *originalScope)
{
    HistogramScope *originalHistogramScope = (HistogramScope*) originalScope;
    HistogramScope *cloneObj = new HistogramScope();

    cloneObj->binWidth = originalHistogramScope->binWidth;
    cloneObj->maxNumberOfBins = originalHistogramScope->maxNumberOfBins;
    cloneObj->m_refreshInterval = originalHistogramScope->m_refreshInterval;

    int histogramSourceCount = originalHistogramScope->m_HistogramSourceConfigs.size();

    for(int i = 0; i < histogramSourceCount; i++)
    {
        Plot2dCurveConfiguration *currentHistogramSourceConf = originalHistogramScope->m_HistogramSourceConfigs.at(i);
        Plot2dCurveConfiguration *newHistogramSourceConf     = new Plot2dCurveConfiguration();

        newHistogramSourceConf->uavObjectName = currentHistogramSourceConf->uavObjectName;
        newHistogramSourceConf->uavFieldName  = currentHistogramSourceConf->uavFieldName;
        newHistogramSourceConf->color         = currentHistogramSourceConf->color;
        newHistogramSourceConf->yScalePower   = currentHistogramSourceConf->yScalePower;
        newHistogramSourceConf->yMeanSamples  = currentHistogramSourceConf->yMeanSamples;
        newHistogramSourceConf->mathFunction  = currentHistogramSourceConf->mathFunction;
        newHistogramSourceConf->yMinimum = currentHistogramSourceConf->yMinimum;
        newHistogramSourceConf->yMaximum = currentHistogramSourceConf->yMaximum;

        cloneObj->m_HistogramSourceConfigs.append(newHistogramSourceConf);
    }

    return cloneObj;
}

void HistogramScope::saveConfiguration(QSettings* qSettings)
{
    //Stop writing XML blocks
    qSettings->beginGroup(QString("plot2d"));
//    qSettings->beginGroup(QString("histogram"));

    qSettings->setValue("plot2dType", HISTOGRAM);
    qSettings->setValue("binWidth", binWidth);
    qSettings->setValue("maxNumberOfBins", maxNumberOfBins);

    int dataSourceCount = m_HistogramSourceConfigs.size();
    qSettings->setValue("dataSourceCount", dataSourceCount);

    // For each curve source in the plot
    for(int i = 0; i < dataSourceCount; i++)
    {
        Plot2dCurveConfiguration *plotCurveConf = m_HistogramSourceConfigs.at(i); //TODO: Understand why this seems to be grabbing i-1
        qSettings->beginGroup(QString("histogramDataSource") + QString().number(i));

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

    //Stop writing XML block
//    qSettings->endGroup();
    qSettings->endGroup();

}


/**
 * @brief HistogramScope::replaceHistogramSource Replaces the list of histogram data sources
 * @param histogramSourceConfigs
 */
void HistogramScope::replaceHistogramDataSource(QList<Plot2dCurveConfiguration*> histogramSourceConfigs)
{
    m_HistogramSourceConfigs.clear();
    m_HistogramSourceConfigs.append(histogramSourceConfigs);
}


/**
 * @brief HistogramScope::loadConfiguration loads the plot configuration into the scope gadget widget
 * @param scopeGadgetWidget
 */
void HistogramScope::loadConfiguration(ScopeGadgetWidget **scopeGadgetWidget)
{
    (*scopeGadgetWidget)->setupHistogramPlot();
    (*scopeGadgetWidget)->setRefreshInterval(m_refreshInterval);

    // Configured each data source
    foreach (Plot2dCurveConfiguration* histogramDataSourceConfig,  m_HistogramSourceConfigs)
    {
        QString uavObjectName = histogramDataSourceConfig->uavObjectName;
        QString uavFieldName = histogramDataSourceConfig->uavFieldName;
        int scale = histogramDataSourceConfig->yScalePower;
        int mean = histogramDataSourceConfig->yMeanSamples;
        QString mathFunction = histogramDataSourceConfig->mathFunction;
        QRgb color = histogramDataSourceConfig->color;

        // TODO: It bothers me that I have to have the ScopeGadgetWidget scopeGadgetWidget in order to call getUavObjectFieldUnits()
        // Get and store the units
        units = (*scopeGadgetWidget)->getUavObjectFieldUnits(uavObjectName, uavFieldName);

        // Create the Qwt histogram plot
        (*scopeGadgetWidget)->addHistogram(
                uavObjectName,
                    uavFieldName,
                    binWidth,
                    maxNumberOfBins,
                    scale,
                    mean,
                    mathFunction,
                    QBrush(QColor(color))
                    );
    }
}


void HistogramScope::setGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page)
{
    //Set the tab widget to 2D
    options_page->tabWidget2d3d->setCurrentWidget(options_page->tabPlot2d);
//    on_tabWidget2d3d_currentIndexChanged(options_page->tabWidget2d3d->currentIndex());

    //Set the plot type
    options_page->cmb2dPlotType->setCurrentIndex(options_page->cmb2dPlotType->findData(getScopeType()));
//    on_cmb2dPlotType_currentIndexChanged(options_page->cmb2dPlotType->currentText());

    //add the configured 2D curves
    options_page->lst2dCurves->clear();  //Clear list first

    foreach (Plot2dCurveConfiguration* dataSource,  m_HistogramSourceConfigs) {
        options_page->spnMaxNumBins->setValue(maxNumberOfBins);
        options_page->spnBinWidth->setValue(binWidth);

        QString uavObjectName = dataSource->uavObjectName;
        QString uavFieldName = dataSource->uavFieldName;
        int scale = dataSource->yScalePower;
        int mean = dataSource->yMeanSamples;
        QString mathFunction = dataSource->mathFunction;
        QVariant varColor = dataSource->color;

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
