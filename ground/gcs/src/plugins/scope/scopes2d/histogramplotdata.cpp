/**
 ******************************************************************************
 *
 * @file       histogramplotplotdata.cpp
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
#include "scopes2d/histogramplotdata.h"
#include "scopes2d/histogramscopeconfig.h"
#include "scopegadgetwidget.h"

#include "qwt/src/qwt.h"
#include "qwt/src/qwt_plot_histogram.h"


#define MAX_NUMBER_OF_INTERVALS 1000


/**
 * @brief HistogramData::HistogramData
 * @param uavObject
 * @param uavField
 * @param binWidth
 * @param numberOfBins
 */
HistogramData::HistogramData(QString uavObject, QString uavField, double binWidth, uint numberOfBins) :
    Plot2dData(uavObject, uavField),
    histogram(0),
    histogramBins(0),
    histogramInterval(0),
    intervalSeriesData(0)
{
    this->binWidth = binWidth;
    this->numberOfBins = numberOfBins;
    scalePower = 1;

    //Create histogram data set
    histogramBins = new QVector<QwtIntervalSample>();
    histogramInterval = new QVector<QwtInterval>();

    // Generate the interval series
    intervalSeriesData = new QwtIntervalSeriesData(*histogramBins);
}


/**
 * @brief HistogramScopeConfig::plotNewData Update plot with new data
 * @param scopeGadgetWidget
 */
void HistogramData::plotNewData(PlotData* plot2dData, ScopeConfig *scopeConfig, ScopeGadgetWidget *scopeGadgetWidget)
{
    Q_UNUSED(plot2dData);
    Q_UNUSED(scopeGadgetWidget);
    Q_UNUSED(scopeConfig);

    //Plot new data
    histogram->setData(intervalSeriesData);
    intervalSeriesData->setSamples(*histogramBins);
}


/**
 * @brief HistogramData::append Appends data to histogram
 * @param obj UAVO with new data
 * @return
 */
bool HistogramData::append(UAVObject* obj)
{

    //Empty histogram data set
    xData->clear();
    yData->clear();

    if (uavObjectName == obj->getName()) {

        //Get the field of interest
        UAVObjectField* field =  obj->getField(uavFieldName);

        //Bad place to do this
        double step = binWidth;
        if (step < 1e-6) //Don't allow step size to be 0.
            step =1e-6;

        if (numberOfBins > MAX_NUMBER_OF_INTERVALS)
            numberOfBins = MAX_NUMBER_OF_INTERVALS;

        if (field) {
            double currentValue = valueAsDouble(obj, field, haveSubField, uavSubFieldName) * pow(10, scalePower);

            // Extend interval, if necessary
            if(!histogramInterval->empty()){
                while (currentValue < histogramInterval->front().minValue()
                       && histogramInterval->size() <= (int) numberOfBins){
                    histogramInterval->prepend(QwtInterval(histogramInterval->front().minValue() - step, histogramInterval->front().minValue()));
                    histogramBins->prepend(QwtIntervalSample(0,histogramInterval->front()));
                }

                while (currentValue > histogramInterval->back().maxValue()
                       && histogramInterval->size() <= (int) numberOfBins){
                    histogramInterval->append(QwtInterval(histogramInterval->back().maxValue(), histogramInterval->back().maxValue() + step));
                    histogramBins->append(QwtIntervalSample(0,histogramInterval->back()));
                }

                // If the histogram reaches its max size, pop one off the end and return
                // This is a graceful way not to lock up the GCS if the bin width
                // is inappropriate, or if there is an extremely distant outlier.
                if (histogramInterval->size() > (int) numberOfBins )
                {
                    histogramBins->pop_back();
                    histogramInterval->pop_back();
                    return false;
                }

                // Test all intervals. This isn't particularly effecient, especially if we have just
                // extended the interval and thus know for sure that the point lies on the extremity.
                // On top of that, some kind of search by bisection would be better.
                for (int i=0; i < histogramInterval->size(); i++ ){
                    if(histogramInterval->at(i).contains(currentValue)){
                        histogramBins->replace(i, QwtIntervalSample(histogramBins->at(i).value + 1, histogramInterval->at(i)));
                        break;
                    }

                }
            }
            else{
                // Create first interval
                double tmp=0;
                if (tmp < currentValue){
                    while (tmp < currentValue){
                        tmp+=step;
                    }
                    histogramInterval->append(QwtInterval(tmp-step, tmp));
                }
                else{
                    while (tmp > step){
                        tmp-=step;
                    }
                    histogramInterval->append(QwtInterval(tmp, tmp+step));
                }

                histogramBins->append(QwtIntervalSample(0,histogramInterval->front()));
            }


            return true;
        }
    }

    return false;
}

/**
 * @brief HistogramScopeConfig::clearPlots Clear all plot data
 */
void HistogramData::clearPlots(PlotData *histogramData)
{
    histogram->detach();

    // Delete data bins
    delete histogramInterval;
    delete histogramBins;

    // Don't delete intervalSeriesData, this is done by the histogram's destructor
    /* delete intervalSeriesData; */

    // Delete histogram (also deletes intervalSeriesData)
    delete histogram;

    delete histogramData;
}
