/**
 ******************************************************************************
 *
 * @file       plotdata.h
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

#ifndef PLOTDATA_H
#define PLOTDATA_H

class ScopeGadgetWidget;
class ScopeConfig;

#include "uavobject.h"

#include "qwt/src/qwt_color_map.h"
#include "qwt/src/qwt_scale_widget.h"

#include <QTimer>
#include <QTime>
#include <QVector>


class PlotData : public QObject
{
    Q_OBJECT
public:
    double valueAsDouble(UAVObject* obj, UAVObjectField* field, bool haveSubField, QString uavSubFieldName);

    //Setter functions
    void setXMinimum(double val){xMinimum=val;}
    virtual void setXMaximum(double val){xMaximum=val;}
    void setYMinimum(double val){yMinimum = val;}
    void setYMaximum(double val){yMaximum = val;}
    void setXWindowSize(double val){m_xWindowSize=val;}
    void setScalePower(int val){scalePower = val;}
    void setMeanSamples(int val){meanSamples = val;}
    void setMathFunction(QString val){mathFunction = val;}

    //Getter functions
    double getXMinimum(){return xMinimum;}
    double getXMaximum(){return xMaximum;}
    double getYMinimum(){return yMinimum;}
    double getYMaximum(){return yMaximum;}
    double getXWindowSize(){return m_xWindowSize;}

    QString getUavoName(){return uavObjectName;}
    QString getUavoFieldName(){return uavFieldName;}
    QString getUavoSubFieldName(){return uavSubFieldName;}
    bool getHaveSubFieldFlag(){return haveSubField;}

    int getScalePower(){return scalePower;}
    int getMeanSamples(){return meanSamples;}
    QString getMathFunction(){return mathFunction;}

    QVector<double>* getXData(){return xData;}
    QVector<double>* getYData(){return yData;}

    virtual bool append(UAVObject* obj) = 0;
    virtual void removeStaleData() = 0;
    virtual void setUpdatedFlagToTrue() = 0;
    virtual bool readAndResetUpdatedFlag() = 0;
    virtual void plotNewData(PlotData *, ScopeConfig *, ScopeGadgetWidget *) = 0;
    virtual void clearPlots(PlotData *) = 0;

    QwtScaleWidget *rightAxis;

protected:
    QVector<double>* xData;    //Data vector for plots
    QVector<double>* yData;    //Used vector for plots

    double m_xWindowSize;
    double xMinimum;
    double xMaximum;
    double yMinimum;
    double yMaximum;

    QString uavObjectName;
    QString uavFieldName;
    QString uavSubFieldName;
    bool haveSubField;

    int scalePower; //This is the power to which each value must be raised
    unsigned int meanSamples;
    QString mathFunction;
    double meanSum;

    double correctionSum;
    int correctionCount;

private:

};

/**
 * @brief The ColorMap class Defines a program-wide colormap
 */
class ColorMap: public QwtLinearColorMap
{
public:
    /**
     * @brief The ColorMapType enum Defines the different type of color maps.
     */
    enum ColorMapType {
        STANDARD,
        JET
    };

    ColorMap(ColorMapType colorMapType = STANDARD):
        QwtLinearColorMap()
    {
        switch (colorMapType){
        case JET:
            createJet();
            break;
        case STANDARD:
        default:
            createStandard();
            break;
        }
    }

    void createJet(){
        // The color interval must be created separately.
        setColorInterval(QColor(0, 0, 30), QColor(0.5*255, 0, 0));

        // Greyscale input values given normalized to 1.
        addColorStop( 0.1, QColor(0.00000*255, 0.00000*255, 0.50000*255));
        addColorStop( 0.2, QColor(0.00000*255, 0.00000*255, 0.94444*255));
        addColorStop( 0.3, QColor(0.00000*255, 0.38889*255, 1.00000*255));
        addColorStop( 0.4, QColor(0.00000*255, 0.83333*255, 1.00000*255));
        addColorStop( 0.5, QColor(0.27778*255, 1.00000*255, 0.72222*255));
        addColorStop( 0.6, QColor(0.72222*255, 1.00000*255, 0.27778*255));
        addColorStop( 0.7, QColor(1.00000*255, 0.83333*255, 0.00000*255));
        addColorStop( 0.8, QColor(1.00000*255, 0.38889*255, 0.00000*255));
        addColorStop( 0.9, QColor(0.94444*255, 0.00000*255, 0.00000*255));

    }

    void createStandard(){
        // The color interval must be created separately.
        setColorInterval(Qt::darkCyan, Qt::red);

        // Greyscale input values given normalized to 1.
        addColorStop( 0.1, Qt::cyan );
        addColorStop( 0.6, Qt::green );
        addColorStop( 0.95, Qt::yellow );

    }
};

#endif // PLOTDATA_H
