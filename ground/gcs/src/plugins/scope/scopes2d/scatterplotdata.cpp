/**
 ******************************************************************************
 *
 * @file       scatterplotdata.cpp
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

#include <QDebug>
#include <math.h>

#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "scopes2d/scatterplotdata.h"
#include "scopes2d/scatterplotscopeconfig.h"
#include "scopegadgetwidget.h"

#include "qwt/src/qwt.h"
#include "qwt/src/qwt_plot.h"
#include "qwt/src/qwt_plot_curve.h"


/**
 * @brief Scatterplot2dScopeConfig::plotNewData Update plot with new data
 * @param scopeGadgetWidget
 */
void TimeSeriesPlotData::plotNewData(PlotData *plot2dData, ScopeConfig *scopeConfig, ScopeGadgetWidget *scopeGadgetWidget)
{
    Q_UNUSED(plot2dData);
    Q_UNUSED(scopeConfig);
    Q_UNUSED(scopeGadgetWidget);

    //Plot new data
    if (readAndResetUpdatedFlag() == true)
        curve->setSamples(*xData, *yData);

    QDateTime NOW = QDateTime::currentDateTime();
    double toTime = NOW.toTime_t();
    toTime += NOW.time().msec() / 1000.0;

    scopeGadgetWidget->setAxisScale(QwtPlot::xBottom, toTime - m_xWindowSize, toTime);
}


/**
 * @brief Scatterplot2dScopeConfig::plotNewData Update plot with new data
 * @param scopeGadgetWidget
 */
void SeriesPlotData::plotNewData(PlotData *plot2dData, ScopeConfig *scopeConfig, ScopeGadgetWidget *scopeGadgetWidget)
{
    Q_UNUSED(plot2dData);
    Q_UNUSED(scopeConfig);
    Q_UNUSED(scopeGadgetWidget);

    //Plot new data
    if (readAndResetUpdatedFlag() == true)
        curve->setSamples(*xData, *yData);
}


/**
 * @brief SeriesPlotData::append Appends data to series plot
 * @param obj UAVO with new data
 * @return
 */
bool SeriesPlotData::append(UAVObject* obj)
{
    if (uavObjectName == obj->getName()) {

        //Get the field of interest
        UAVObjectField* field =  obj->getField(uavFieldName);

        if (field) {

            double currentValue = valueAsDouble(obj, field, haveSubField, uavSubFieldName) * pow(10, scalePower);

            //Perform scope math, if necessary
            if (mathFunction  == "Boxcar average" || mathFunction  == "Standard deviation"){
                //Put the new value at the front
                yDataHistory->append( currentValue );

                // calculate average value
                meanSum += currentValue;
                if(yDataHistory->size() > (int)meanSamples) {
                    meanSum -= yDataHistory->first();
                    yDataHistory->pop_front();
                }

                // make sure to correct the sum every meanSamples steps to prevent it
                // from running away due to floating point rounding errors
                correctionSum+=currentValue;
                if (++correctionCount >= (int)meanSamples) {
                    meanSum = correctionSum;
                    correctionSum = 0.0f;
                    correctionCount = 0;
                }

                double boxcarAvg=meanSum/yDataHistory->size();

                if ( mathFunction  == "Standard deviation" ){
                    //Calculate square of sample standard deviation, with Bessel's correction
                    double stdSum=0;
                    for (int i=0; i < yDataHistory->size(); i++){
                        stdSum+= pow(yDataHistory->at(i)- boxcarAvg,2)/(meanSamples-1);
                    }
                    yData->append(sqrt(stdSum));
                }
                else  {
                    yData->append(boxcarAvg);
                }
            }
            else{
                yData->append( currentValue );
            }

            if (yData->size() > getXWindowSize()) { //If new data overflows the window, remove old data...
                yData->pop_front();
            } else //...otherwise, add a new y point at position xData
                xData->insert(xData->size(), xData->size());

            return true;
        }
    }

    return false;
}


/**
 * @brief TimeSeriesPlotData::append Appends data to time series data
 * @param obj UAVO with new data
 * @return
 */
bool TimeSeriesPlotData::append(UAVObject* obj)
{
    if (uavObjectName == obj->getName()) {
        //Get the field of interest
        UAVObjectField* field =  obj->getField(uavFieldName);

        if (field) {
            QDateTime NOW = QDateTime::currentDateTime(); //THINK ABOUT REIMPLEMENTING THIS TO SHOW UAVO TIME, NOT SYSTEM TIME
            double currentValue = valueAsDouble(obj, field, haveSubField, uavSubFieldName) * pow(10, scalePower);

            //Perform scope math, if necessary
            if (mathFunction  == "Boxcar average" || mathFunction  == "Standard deviation"){
                //Put the new value at the back
                yDataHistory->append( currentValue );

                // calculate average value
                meanSum += currentValue;
                if(yDataHistory->size() > (int)meanSamples) {
                    meanSum -= yDataHistory->first();
                    yDataHistory->pop_front();
                }
                // make sure to correct the sum every meanSamples steps to prevent it
                // from running away due to floating point rounding errors
                correctionSum+=currentValue;
                if (++correctionCount >= (int)meanSamples) {
                    meanSum = correctionSum;
                    correctionSum = 0.0f;
                    correctionCount = 0;
                }

                double boxcarAvg=meanSum/yDataHistory->size();

                if ( mathFunction  == "Standard deviation" ){
                    //Calculate square of sample standard deviation, with Bessel's correction
                    double stdSum=0;
                    for (int i=0; i < yDataHistory->size(); i++){
                        stdSum+= pow(yDataHistory->at(i)- boxcarAvg,2)/(meanSamples-1);
                    }
                    yData->append(sqrt(stdSum));
                }
                else  {
                    yData->append(boxcarAvg);
                }
            }
            else{
                yData->append( currentValue );
            }

            double valueX = NOW.toTime_t() + NOW.time().msec() / 1000.0;
            xData->append(valueX);

            //Remove stale data
            removeStaleData();

            return true;
        }
    }

    return false;
}


/**
 * @brief TimeSeriesPlotData::removeStaleData Removes stale data from time series plot
 */
void TimeSeriesPlotData::removeStaleData()
{
    double newestValue;
    double oldestValue;

    while (1) {
        if (xData->size() == 0)
            break;

        newestValue = xData->last();
        oldestValue = xData->first();

        if (newestValue - oldestValue > getXWindowSize()) {
            yData->pop_front();
            xData->pop_front();
        } else
            break;
    }
}


/**
 * @brief TimeSeriesPlotData::removeStaleDataTimeout On timer timeout, removes data that can no longer be seen on axes.
 */
void TimeSeriesPlotData::removeStaleDataTimeout()
{
    removeStaleData();
}


/**
 * @brief ScatterplotData::clearPlots Clear all plot data
 */
void ScatterplotData::clearPlots(PlotData *scatterplotData)
{
    curve->detach();

    delete curve;
    delete scatterplotData;
}
