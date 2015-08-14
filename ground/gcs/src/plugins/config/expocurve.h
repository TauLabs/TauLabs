
/**
 ******************************************************************************
 * @file       expocurve.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Visualize the expo seettings
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

#ifndef EXPOCURVE_H
#define EXPOCURVE_H


#include <QWidget>
#define QWT_DLL
#include "qwt/src/qwt.h"
#include "qwt/src/qwt_plot.h"
#include "qwt/src/qwt_plot_curve.h"
#include "qwt/src/qwt_scale_draw.h"
#include "qwt/src/qwt_scale_widget.h"
#include "qwt/src/qwt_plot_grid.h"
#include "qwt/src/qwt_legend.h"
#include "qwt/src/qwt_legend_label.h"
#include "qwt/src/qwt_plot_marker.h"
#include "qwt/src/qwt_symbol.h"

class ExpoCurve : public QwtPlot
{
    Q_OBJECT
public:
    explicit ExpoCurve(QWidget *parent = 0);

    typedef struct ExpoPlotElements {
      QwtPlotCurve Curve;
      QwtPlotMarker Mark;
      QwtPlotMarker Mark_;
      QwtPlotCurve Curve2;
      QwtPlotMarker Mark2;
      QwtPlotMarker Mark2_;
    } ExpoPlotElements_t;

    enum axis_mode {Y_Left, Y_Right};
    enum label_mode {RateCurve, AttitudeCurve, HorizonCurve};

    //! Set label for the stick channels
    void init(label_mode lbl_mode, int h_transistion);

    //! Show expo data for one of the stick channels
    void plotData(int value, int max, ExpoPlotElements_t &plot_elements, axis_mode mode);

public slots:

    //! Show expo data for roll
    void plotDataRoll(double value, int max, axis_mode mode);

    //! Show expo data for pitch
    void plotDataPitch(double value, int max, axis_mode mode);

    //! Show expo data for yaw
    void plotDataYaw(double value, int max, axis_mode mode);

    //! Show/Hide a expo curve and markers
    void showCurve(const QVariant & itemInfo, bool on, int index);

signals:

public slots:

private:

    int steps;
    int horizon_transition;
    int curve_cnt;
    double *x_data;
    double *y_data;

    ExpoPlotElements_t roll_elements;
    ExpoPlotElements_t pitch_elements;
    ExpoPlotElements_t yaw_elements;

    //! Inverse expo function
    double invers_expo3(double y, int g);
};

#endif // EXPOCURVE_H
