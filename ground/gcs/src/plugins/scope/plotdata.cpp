/**
 ******************************************************************************
 *
 * @file       plotdata.cpp
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

#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"


#include "scopes2d/histogramdata.h"
#include "plotdata2d.h"
#include "plotdata3d.h"
#include <math.h>
#include <QDebug>

#define MAX_NUMBER_OF_INTERVALS 1000

Plot2dData::Plot2dData(QString p_uavObject, QString p_uavFieldName):
    yDataHistory(0),
    dataUpdated(false)
{
    uavObjectName = p_uavObject;

    if(p_uavFieldName.contains("-")) //For fields with multiple indices, '-' followed by an index indicates which one
    {
        QStringList fieldSubfield = p_uavFieldName.split("-", QString::SkipEmptyParts);
        uavFieldName = fieldSubfield.at(0);
        uavSubFieldName = fieldSubfield.at(1);
        haveSubField = true;
    }
    else
    {
        uavFieldName =  p_uavFieldName;
        haveSubField = false;
    }

    xData = new QVector<double>();
    yData = new QVector<double>();
    yDataHistory = new QVector<double>();

    scalePower = 0;
    meanSamples = 1;
    meanSum = 0.0f;
    correctionSum = 0.0f;
    correctionCount = 0;
    yMinimum = 0;
    yMaximum = 120;

    m_xWindowSize = 0;
}


Plot3dData::Plot3dData(QString p_uavObject, QString p_uavFieldName):
    dataUpdated(false)
{
    uavObjectName = p_uavObject;

    //TODO: This needs a comment here. How can a `-` appear in a UAVO field name? Is this automatic in certain instances, or is it user-defined?
    if(p_uavFieldName.contains("-"))
    {
        QStringList fieldSubfield = p_uavFieldName.split("-", QString::SkipEmptyParts);
        uavFieldName = fieldSubfield.at(0);
        uavSubFieldName = fieldSubfield.at(1);
        haveSubField = true;
    }
    else
    {
        uavFieldName =  p_uavFieldName;
        haveSubField = false;
    }

    xData = new QVector<double>();
    yData = new QVector<double>();
    zData = new QVector<double>();
    zDataHistory = new QVector<double>();
    timeDataHistory = new QVector<double>();

    curve = 0;
    scalePower = 0;
    meanSamples = 1;
    meanSum = 0.0f;
    correctionSum = 0.0f;
    correctionCount = 0;
    xMinimum = 0;
    xMaximum = 16;
    yMinimum = 0;
    yMaximum = 60;
    zMinimum = 0;
    zMaximum = 100;

}


Plot2dData::~Plot2dData()
{
    if (xData != NULL)
        delete xData;
    if (yData != NULL)
        delete yData;
    if (yDataHistory != NULL)
        delete yDataHistory;
}


Plot3dData::~Plot3dData()
{
    if (xData != NULL)
        delete xData;
    if (yData != NULL)
        delete yData;
    if (zData != NULL)
        delete zData;
    if (zDataHistory != NULL)
        delete zDataHistory;
    if (timeDataHistory != NULL)
        delete timeDataHistory;
}


double valueAsDouble(UAVObject* obj, UAVObjectField* field, bool haveSubField, QString uavSubFieldName)
{
    Q_UNUSED(obj);
    QVariant value;

    if(haveSubField){
        int indexOfSubField = field->getElementNames().indexOf(QRegExp(uavSubFieldName, Qt::CaseSensitive, QRegExp::FixedString));
        value = field->getValue(indexOfSubField);
    }else
        value = field->getValue();

    return value.toDouble();
}


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
                if(yDataHistory->size() > meanSamples) {
                    meanSum -= yDataHistory->first();
                    yDataHistory->pop_front();
                }

                // make sure to correct the sum every meanSamples steps to prevent it
                // from running away due to floating point rounding errors
                correctionSum+=currentValue;
                if (++correctionCount >= meanSamples) {
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


bool SpectrogramData::append(UAVObject* multiObj)
{
    QDateTime NOW = QDateTime::currentDateTime(); //TODO: This should show UAVO time and not system time

    // Check to make sure it's the correct UAVO
    if (uavObjectName == multiObj->getName()) {

        // Only run on UAVOs that have multiple instances
        if (multiObj->isSingleInstance())
            return false;

        //Instantiate object manager
        UAVObjectManager *objManager;

        ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
        Q_ASSERT(pm != NULL);
        objManager = pm->getObject<UAVObjectManager>();
        Q_ASSERT(objManager != NULL);


        // Get list of object instances
        QList<UAVObject*> list = objManager->getObjectInstances(multiObj->getName());

        // Remove a row's worth of data.
        unsigned int spectrogramWidth = list.length();

        // Check that there is a full window worth of data. While GCS is starting up, the size of
        // multiple instance UAVOs is 1, so it's possible for spurious data to come in before
        // the flight controller board has had time to initialize the UAVO size.
        if (spectrogramWidth != windowWidth){
            qDebug() << "Incomplete data set in" << multiObj->getName() << "." << uavFieldName <<  "spectrogram: " << spectrogramWidth << " samples provided, but expected " << windowWidth;
            return false;
        }

        //Initialize vector where we will read out an entire row of multiple instance UAVO
        QVector<double> values;

        timeDataHistory->append(NOW.toTime_t() + NOW.time().msec() / 1000.0);
        UAVObjectField* multiField =  multiObj->getField(uavFieldName);
        Q_ASSERT(multiField);
        if (multiField ) {

            // Get the field of interest
            foreach (UAVObject *obj, list) {
                UAVObjectField* field =  obj->getField(uavFieldName);

                double currentValue = valueAsDouble(obj, field, haveSubField, uavSubFieldName) * pow(10, scalePower);

                double vecVal = currentValue;
                //Normally some math would go here, modifying vecVal before appending it to values
                // .
                // .
                // .


                // Second to last step, see if autoscale is turned on and if the value exceeds the maximum for the scope.
                if ( getZMaximum() == 0 &&  vecVal > rasterData->interval(Qt::ZAxis).maxValue()){
                    // Change scope maximum and color depth
                    rasterData->setInterval(Qt::ZAxis, QwtInterval(0, vecVal) );
                    autoscaleValueUpdated = vecVal;
                }
                // Last step, assign value to vector
                values += vecVal;
            }

            while (timeDataHistory->back() - timeDataHistory->front() > timeHorizon){
                timeDataHistory->pop_front();
                zDataHistory->remove(0, fminl(spectrogramWidth, zDataHistory->size()));
            }

            // Doublecheck that there are the right number of samples. This can occur if the "field" assert fails
            if(values.size() == (int) windowWidth){
                *zDataHistory << values;
            }

            return true;
        }
    }

    return false;
}


bool TimeSeriesPlotData::append(UAVObject* obj)
{
    if (uavObjectName == obj->getName()) {
        //Get the field of interest
        UAVObjectField* field =  obj->getField(uavFieldName);
        //qDebug() << "uavObject: " << uavObject << ", uavField: " << uavFieldName;

        if (field) {
            QDateTime NOW = QDateTime::currentDateTime(); //THINK ABOUT REIMPLEMENTING THIS TO SHOW UAVO TIME, NOT SYSTEM TIME
            double currentValue = valueAsDouble(obj, field, haveSubField, uavSubFieldName) * pow(10, scalePower);

            //Perform scope math, if necessary
            if (mathFunction  == "Boxcar average" || mathFunction  == "Standard deviation"){
                //Put the new value at the back
                yDataHistory->append( currentValue );

                // calculate average value
                meanSum += currentValue;
                if(yDataHistory->size() > meanSamples) {
                    meanSum -= yDataHistory->first();
                    yDataHistory->pop_front();
                }
                // make sure to correct the sum every meanSamples steps to prevent it
                // from running away due to floating point rounding errors
                correctionSum+=currentValue;
                if (++correctionCount >= meanSamples) {
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

void TimeSeriesPlotData::removeStaleDataTimeout()
{
    removeStaleData();
}

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
