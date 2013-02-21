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

#include "scopes2d/scatterplotscopeconfig.h"


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

//TODO: This should probably be a constructor, too.
void Scatterplot2dScope::clone(ScopesGeneric *originalScope)
{
//    this->parent() = new Scatterplot2dScope();

    Scatterplot2dScope *originalScatterplot2dScope = (Scatterplot2dScope*) originalScope;

    m_refreshInterval = originalScatterplot2dScope->m_refreshInterval;
    timeHorizon = originalScatterplot2dScope->timeHorizon;
    scatterplot2dType = originalScatterplot2dScope->scatterplot2dType;

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

        m_scatterplotSourceConfigs.append(newScatterplotSourceConf);
    }

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
        (*scopeGadgetWidget)->setupSeriesPlot();
        break;
    case TIMESERIES2D:
        (*scopeGadgetWidget)->setupTimeSeriesPlot();
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
        int scale = plotCurveConfig->yScalePower;
        int mean = plotCurveConfig->yMeanSamples;
        QString mathFunction = plotCurveConfig->mathFunction;
        QRgb color = plotCurveConfig->color;

        // Create the Qwt curve plot
        (*scopeGadgetWidget)->add2dCurvePlot(
                    uavObjectName,
                    uavFieldName,
                    scale,
                    mean,
                    mathFunction,
                    QPen(  QBrush(QColor(color),Qt::SolidPattern),
                       (qreal)1,
                       Qt::SolidLine,
                       Qt::SquareCap,
                       Qt::BevelJoin)
                    );
    }
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

