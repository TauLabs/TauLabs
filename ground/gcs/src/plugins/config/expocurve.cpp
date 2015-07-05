/**
 ******************************************************************************
 * @file       txpocurve.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Visualize the expo settings
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

#include "expocurve.h"

ExpoCurve::ExpoCurve(QWidget *parent) :
    QwtPlot(parent)
{
    setMouseTracking(true);

    setMinimumSize(64, 64);
    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    //setCanvasBackground(QColor(64, 64, 64));

    //Add grid lines
    QwtPlotGrid *grid = new QwtPlotGrid;
    grid->setMajorPen(QPen(Qt::gray, 0, Qt::DashLine));
    grid->setMinorPen(QPen(Qt::lightGray, 0, Qt::DotLine));
    grid->setPen(QPen(Qt::darkGray, 1, Qt::DotLine));
    grid->attach(this);

    roll_elements.Curve.setRenderHint(QwtPlotCurve::RenderAntialiased);
    roll_elements.Curve.setPen(QPen(QColor(Qt::blue), 1.0));
    roll_elements.Curve.attach(this);

    pitch_elements.Curve.setRenderHint(QwtPlotCurve::RenderAntialiased);
    pitch_elements.Curve.setPen(QPen(QColor(Qt::red), 1.0));
    pitch_elements.Curve.attach(this);

    yaw_elements.Curve.setRenderHint(QwtPlotCurve::RenderAntialiased);
    yaw_elements.Curve.setPen(QPen(QColor(Qt::green), 1.0));
    yaw_elements.Curve.attach(this);

    roll_elements.Curve2.setRenderHint(QwtPlotCurve::RenderAntialiased);
    roll_elements.Curve2.setPen(QPen(QColor(Qt::darkBlue), 1.0, Qt::DashLine));
    roll_elements.Curve2.setYAxis(QwtPlot::yRight);

    pitch_elements.Curve2.setRenderHint(QwtPlotCurve::RenderAntialiased);
    pitch_elements.Curve2.setPen(QPen(QColor(Qt::darkRed), 1.0, Qt::DashLine));
    pitch_elements.Curve2.setYAxis(QwtPlot::yRight);

    yaw_elements.Curve2.setRenderHint(QwtPlotCurve::RenderAntialiased);
    yaw_elements.Curve2.setPen(QPen(QColor(Qt::darkGreen), 1.0, Qt::DashLine));
    yaw_elements.Curve2.setYAxis(QwtPlot::yRight);

    // legend
    // Show a legend at the top
    QwtLegend *m_legend = new QwtLegend(this);
    m_legend->setDefaultItemMode(QwtLegendData::Checkable);
    m_legend->setFrameStyle(QFrame::Box | QFrame::Sunken);
    m_legend->setToolTip(tr("Click legend to show/hide expo curve"));

    // connect signal when clicked on legend entry to function that shows/hides the curve
    connect(m_legend, SIGNAL(checked(const QVariant &, bool, int)), this, SLOT(showCurve(QVariant, bool, int)));

    QPalette pal = m_legend->palette();
    pal.setColor(m_legend->backgroundRole(), QColor(100, 100, 100));	// background colour
    pal.setColor(QPalette::Text, QColor(0, 0, 0));			// text colour
    m_legend->setPalette(pal);

    insertLegend(m_legend, QwtPlot::TopLegend);
    // QwtPlot::insertLegend() changes the max columns attribute, so you have to set it to the desired number after the statement
    m_legend->setMaxColumns(3);


    steps = 1000;
    x_data =  new double[steps];
    y_data =  new double[steps];

    double step = 2 * 1.0 / (steps - 1);
    for (int i = 0; i < steps; i++) {
        x_data[i] = (i * step) - 1.0;
    }

    horizon_transition = 0;


    //marker for horizon transition value
    QwtSymbol *sym1 = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::blue), QPen(Qt::blue), QSize(7, 7));
    roll_elements.Mark.setSymbol(sym1);
    QwtSymbol *sym1_ = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::blue), QPen(Qt::blue), QSize(7, 7));
    roll_elements.Mark_.setSymbol(sym1_);
    QwtSymbol *sym1_2 = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::darkBlue), QPen(Qt::darkBlue), QSize(7, 7));
    roll_elements.Mark2.setSymbol(sym1_2);
    QwtSymbol *sym1_2_ = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::darkBlue), QPen(Qt::darkBlue), QSize(7, 7));
    roll_elements.Mark2_.setSymbol(sym1_2_);
    roll_elements.Mark2.setYAxis(QwtPlot::yRight);
    roll_elements.Mark2_.setYAxis(QwtPlot::yRight);

    QwtSymbol *sym2 = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::red), QPen(Qt::red), QSize(7, 7));
    pitch_elements.Mark.setSymbol(sym2);
    QwtSymbol *sym2_ = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::red), QPen(Qt::red), QSize(7, 7));
    pitch_elements.Mark_.setSymbol(sym2_);
    QwtSymbol *sym2_2 = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::darkRed), QPen(Qt::darkRed), QSize(7, 7));
    pitch_elements.Mark2.setSymbol(sym2_2);
    QwtSymbol *sym2_2_ = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::darkRed), QPen(Qt::darkRed), QSize(7, 7));
    pitch_elements.Mark2_.setSymbol(sym2_2_);
    pitch_elements.Mark2.setYAxis(QwtPlot::yRight);
    pitch_elements.Mark2_.setYAxis(QwtPlot::yRight);

    QwtSymbol *sym3 = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::green), QPen(Qt::green), QSize(7, 7));
    yaw_elements.Mark.setSymbol(sym3);
    QwtSymbol *sym3_ = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::green), QPen(Qt::green), QSize(7, 7));
    yaw_elements.Mark_.setSymbol(sym3_);
    QwtSymbol *sym3_2 = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::darkGreen), QPen(Qt::darkGreen), QSize(7, 7));
    yaw_elements.Mark2.setSymbol(sym3_2);
    QwtSymbol *sym3_2_ = new QwtSymbol(QwtSymbol::Star1, QBrush(Qt::darkGreen), QPen(Qt::darkGreen), QSize(7, 7));
    yaw_elements.Mark2_.setSymbol(sym3_2_);
    yaw_elements.Mark2.setYAxis(QwtPlot::yRight);
    yaw_elements.Mark2_.setYAxis(QwtPlot::yRight);
}

/**
 * @brief ExpoCurve::init Init labels, titels, horizin transition,...
 * @param lbl_mode Chose the mode of this widget; RateCurve: for rate mode, HorizonCurve: for horizon mode
 * @param horizon_transitions value for the horizon transition markers in the plot; 0: disabled, >0: horizon transitions in % horizon (should be the same as defined in /flight/Modules/Stabilization/stabilization.c)
 */
void ExpoCurve::init(label_mode lbl_mode, int h_transistion)
{
    //  setup of the axis title
    QwtText axis_title;

    // get the default font
    QFont axis_title_font = this->axisTitle(yLeft).font();
    // and only change the font size
    axis_title_font.setPointSize(10);
    axis_title.setFont(axis_title_font);

    this->enableAxis(QwtPlot::yRight);

    axis_title.setText(tr(" normalized stick input"));
    this->setAxisTitle(QwtPlot::xBottom, axis_title);

    switch (lbl_mode)
    {
        case RateCurve:
            roll_elements.Curve.setTitle(tr("Roll rate (deg/s)"));
            pitch_elements.Curve.setTitle(tr("Pitch rate (deg/s)"));
            yaw_elements.Curve.setTitle(tr("Yaw rate (deg/s)"));

            axis_title.setText(tr("rate (deg/s)"));
            this->setAxisTitle(QwtPlot::yRight, axis_title);
            this->setAxisTitle(QwtPlot::yLeft, axis_title);
            curve_cnt = 1;

            break;
        case HorizonCurve:
            roll_elements.Curve.setTitle(tr("Roll angle (deg)"));
            pitch_elements.Curve.setTitle(tr("Pitch angle (deg)"));
            yaw_elements.Curve.setTitle(tr("Yaw angle (deg)"));
            roll_elements.Curve2.setTitle(tr("Roll rate (deg/s)"));
            pitch_elements.Curve2.setTitle(tr("Pitch rate (deg/s)"));
            yaw_elements.Curve2.setTitle(tr("Yaw rate (deg/s)"));
            curve_cnt = 2;

            axis_title.setText(tr("horizon angle (deg)"));
            this->setAxisTitle(QwtPlot::yLeft, axis_title);
            axis_title.setText(tr("horizon rate (deg/s)"));
            this->setAxisTitle(QwtPlot::yRight, axis_title);

            this->setToolTip(tr("This plot shows data only for the Horizon mode, not the Attitude mode.<br><br>For each axis there are 2 curves.<br>One for the 'Horizon Attitude' part (horizon angle) and in darker color one for the 'Horizon Rate' part (horizon rate).<br><br>The markers on the curves show the threshold point:<br><br>- For stick inputs above its purely the 'Horizon Rate' mode with expo scaling by the 'Expo Horizon'<br><br>- For stick inputs below there is a dynamic transistion from 'Horizon Attitude' to 'Horizon Rate' mode."));
            break;
    }

    horizon_transition = h_transistion;

    if (horizon_transition != 0) {
        if ( (curve_cnt == 1) || (curve_cnt == 2) ) {
            roll_elements.Mark.attach(this);
            roll_elements.Mark_.attach(this);
            pitch_elements.Mark.attach(this);
            pitch_elements.Mark_.attach(this);
            yaw_elements.Mark.attach(this);
            yaw_elements.Mark_.attach(this);
        }

        if ( (curve_cnt == 2) ) {
            //this->enableAxis(QwtPlot::yRight);
            roll_elements.Curve2.attach(this);
            pitch_elements.Curve2.attach(this);
            yaw_elements.Curve2.attach(this);

            roll_elements.Mark2.attach(this);
            roll_elements.Mark2_.attach(this);
            pitch_elements.Mark2.attach(this);
            pitch_elements.Mark2_.attach(this);
            yaw_elements.Mark2.attach(this);
            yaw_elements.Mark2_.attach(this);
        }
    }
}

/**
 * @brief ExpoCurve::plotData Show expo data for one of the stick channels
 * @param value The expo coefficient; sets the exponential amount [0,100]
 * @param curve The curve that has to be plot (roll,nick,yaw)
 * @param mode  The mode chooses wich y-axis is used: Y_Right or Y_Left
 *
 * The math here is copied/ the same as in the expo3() function in /flight/Libraries/math/misc_math.c
 * Please be aware of changes that are made there.
 */
void ExpoCurve::plotData(int value, int max, ExpoPlotElements_t &plot_elements, axis_mode mode)
{
    double marker_x;
    double marker_y;

    for (int i = 0; i < steps; i++) {
        y_data[i] = max * (x_data[i]  * ((100 - value) / 100.0) + pow(x_data[i], 3) * (value / 100.0));
    }

    if (mode == Y_Left) {
        plot_elements.Curve.setSamples(x_data, y_data, steps);
        plot_elements.Curve.show();
    }
    else if (mode == Y_Right) {
        plot_elements.Curve2.setSamples(x_data, y_data, steps);
        plot_elements.Curve2.show();
    }

    if (horizon_transition != 0) {
        marker_x = invers_expo3((double) (horizon_transition / 100.0), value);
        marker_y = max * horizon_transition / 100;

        // additional scaling (*0.985) of positive x marker position for better visual fit with the curve
        if (mode == Y_Left) {
            plot_elements.Mark.setValue(marker_x * 0.985, marker_y);
            plot_elements.Mark_.setValue(-1 * marker_x, -1 * marker_y);
        }
        else if (mode == Y_Right) {
            plot_elements.Mark2.setValue(marker_x * 0.985, marker_y);
            plot_elements.Mark2_.setValue(-1 * marker_x, -1 * marker_y);
        }
    }

    this->replot();

    if( curve_cnt == 1) {
        this->setAxisScaleDiv(yRight, this->axisScaleDiv(yLeft));
        this->replot();
    }

}

/**
 * @brief ExpoCurve::plotDataRoll public function to plot expo data for roll axis
 * @param value The expo coefficient; sets the exponential amount [0,100]
 * @param max   The max. scaling for the axis in physical units
 * @param mode  The mode chooses wich y-axis is used: Y_Right or Y_Left
 */
void ExpoCurve::plotDataRoll(double value, int max, axis_mode mode)
{
    plotData((int)value, max, this->roll_elements, mode);
}

/**
 * @brief ExpoCurve::plotDataPitch public function to plot expo data for pitch axis
 * @param value The expo coefficient; sets the exponential amount [0,100]
 * @param max   The max. scaling for the axis in physical units
 * @param mode  The mode chooses wich y-axis is used: Y_Right or Y_Left
 */
void ExpoCurve::plotDataPitch(double value, int max, axis_mode mode)
{
    plotData((int)value, max, this->pitch_elements, mode);
}

/**
 * @brief ExpoCurve::plotDataYaw public function to plot expo data for yaw axis
 * @param value The expo coefficient; sets the exponential amount [0,100]
 * @param max   The max. scaling for the axis in physical units
 * @param mode  The mode chooses wich y-axis is used: Y_Right or Y_Left
 */
void ExpoCurve::plotDataYaw(double value, int max, axis_mode mode)
{
    plotData((int)value, max, this->yaw_elements, mode);
}

/**
 * @brief ExpoCurve::invers_expo3 Calcs the invers expo3 data
 * @param y The expo output data from [-1,1]
 * @param g Sets the exponential amount [0,100]
 *
 * The math here is done with help of: http://www.wolframalpha.com/input/?i=Solve%5By%3D%28%28g%2F100%29*x%5E3%2B%28%28100-g%29%2F100%29*x%29%2Cx%5D
 * e.g. y=0.85 expo = 50% : http://www.wolframalpha.com/input/?i=Solve%5B0.85%3D%28%2850%2F100%29*x%5E3%2B%28%28100-50%29%2F100%29*x%29%2Cx%5D
 * It is done for the expo3 implementation from /flight. Please be aware of changes that are made there.
 */
double ExpoCurve::invers_expo3(double y, int g)
{
    double temp1;
    double temp2;
    if (g > 0) {
        //double temp1 = pow((2700*pow(g,2)*y + sqrt(7290000*pow(g,4)*pow(y,2)+108*pow((100-g),3)*pow(g,3))),(1.0/3.0));
        temp1 = 108 * pow((100-g), 3) * pow(g, 3);
        temp1 = sqrt(7290000 * pow(g, 4) * pow(y, 2) + temp1);
        temp1 = 2700 * pow(g, 2) * y + temp1;
        temp1 = pow(temp1, (1.0 / 3.0));

        temp2 = pow(2, (1.0 / 3.0));

        return (temp1 / (3 * g * temp2) - ((100 - g) * temp2) / temp1);
    }
    else {
        return y;
    }
}


/**
 * @brief ExpoCurve::showCurve The Slot function to show/hide a curve and the corresponding markers. Called from a "checked" Signal
 * @param itemInfo Info for the item of the selected legend label
 * @param on       True when the legend label is checked
 * @param index    Index of the legend label in the list of widgets that are associated with the plot item; but not used here
 */
void ExpoCurve::showCurve(const QVariant & itemInfo, bool on, int index)
{
    Q_UNUSED(index);
    QwtPlotItem * item = QwtPlot::infoToItem(itemInfo);
    if (item) {
        item->setVisible(!on);

        if (item == &this->roll_elements.Curve) {
            this->roll_elements.Mark.setVisible(!on);
            this->roll_elements.Mark_.setVisible(!on);
        }
        else if (item == &this->pitch_elements.Curve) {
            this->pitch_elements.Mark.setVisible(!on);
            this->pitch_elements.Mark_.setVisible(!on);
        }
        else if (item == &this->yaw_elements.Curve) {
            this->yaw_elements.Mark.setVisible(!on);
            this->yaw_elements.Mark_.setVisible(!on);
        }
        else if (item == &this->roll_elements.Curve2) {
            this->roll_elements.Mark2.setVisible(!on);
            this->roll_elements.Mark2_.setVisible(!on);
        }
        else if (item == &this->pitch_elements.Curve2) {
            this->pitch_elements.Mark2.setVisible(!on);
            this->pitch_elements.Mark2_.setVisible(!on);
        }
        else if (item == &this->yaw_elements.Curve2) {
            this->yaw_elements.Mark2.setVisible(!on);
            this->yaw_elements.Mark2_.setVisible(!on);
        }
    }

    replot();
}
