/**
 ******************************************************************************
 * @file       tempcompcurve.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Display the results of temperature compensation
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

#include "tempcompcurve.h"

TempCompCurve::TempCompCurve(QWidget *parent) :
    QwtPlot(parent), dataCurve(NULL), fitCurve(NULL)
{
    setMouseTracking(true);

    setMinimumSize(64, 64);
    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    setCanvasBackground(QColor(64, 64, 64));

    //Add grid lines
    QwtPlotGrid *grid = new QwtPlotGrid;
    grid->setMajPen(QPen(Qt::gray, 0, Qt::DashLine));
    grid->setMinPen(QPen(Qt::lightGray, 0, Qt::DotLine));
    grid->setPen(QPen(Qt::darkGray, 1, Qt::DotLine));
    grid->attach(this);
}

/**
 * @brief TempCompCurve::plotData Visualize the measured data
 * @param temp The set of temperature measurements
 * @param gyro The set of gyro measurements
 */
void TempCompCurve::plotData(QList<double> temp, QList<double> gyro, QList <double> coeff)
{
    // TODO: Keep the curves and free them in the destructors
    const int STEPS = 100;

    points.clear();
    fit.clear();

    clearCurves();

    double min = temp[0];
    double max = temp[0];
    for (int i = 0; i < temp.size(); i++) {
        points.append(QPointF(temp[i],gyro[i]));
        min = qMin(min, temp[i]);
        max = qMax(max, temp[i]);
    }

    double step = (max - min) / STEPS;
    for (int i = 0; i < STEPS; i++) {
        double t = min + step * i;
        double f = coeff[0] + coeff[1] * t + coeff[2] * pow(t,2) + coeff[3] * pow(t,3);
        fit.append(QPointF(t, f));
    }

    // Plot the raw data points
    QPen pen = QPen(Qt::DotLine);
    pen.setColor(QColor(244, 244, 244));

    dataCurve = new QwtPlotCurve("Gyro");
    dataCurve->setPen(pen);
    dataCurve->setSamples(this->points);
    dataCurve->attach(this);

    // Plot the fit
    pen.setStyle(Qt::SolidLine);
    pen.setColor(QColor(0,255,0));

    fitCurve = new QwtPlotCurve("Fit");
    fitCurve->setPen(pen);
    fitCurve->setSamples(fit);
    fitCurve->attach(this);

    replot();
}

/**
 * @brief TempCompCurve::clearCurves Remove any previous curves and delete them
 */
void TempCompCurve::clearCurves()
{
    // Remove previous curves
    if (dataCurve) {
        dataCurve->detach();
        delete dataCurve;
        dataCurve = NULL;
    }
    if (fitCurve) {
        fitCurve->detach();
        delete fitCurve;
        fitCurve = NULL;
    }
}
