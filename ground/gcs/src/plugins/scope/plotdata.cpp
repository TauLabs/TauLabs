/**
 ******************************************************************************
 *
 * @file       plotdata.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
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

#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"

#include "scopes2d/plotdata2d.h"
#include "scopes3d/plotdata3d.h"

#include <math.h>
#include <QDebug>

/**
 * @brief Plot2dData::Plot2dData Default 2d constructor
 * @param p_uavObject The plotted UAVO name
 * @param p_uavFieldName The plotted UAVO field name
 */
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


/**
 * @brief Plot3dData::Plot3dData Default 3d constructor
 * @param p_uavObject The plotted UAVO name
 * @param p_uavFieldName The plotted UAVO field name
 */
Plot3dData::Plot3dData(QString p_uavObject, QString p_uavFieldName):
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
    zData = new QVector<double>();
    zDataHistory = new QVector<double>();
    timeDataHistory = new QVector<double>();

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


/**
 * @brief valueAsDouble Fetch the value from the UAVO and return it as a double
 * @param obj UAVO
 * @param field UAVO field
 * @param haveSubField TRUE if UAVO has subfield. FALSE if not.
 * @param uavSubFieldName UAVO subfield, if it exists
 * @return
 */
double PlotData::valueAsDouble(UAVObject* obj, UAVObjectField* field, bool haveSubField, QString uavSubFieldName)
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
