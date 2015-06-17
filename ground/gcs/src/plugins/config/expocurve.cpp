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

    rollElements.Curve.setRenderHint(QwtPlotCurve::RenderAntialiased);
    rollElements.Curve.setPen(QPen(QColor(Qt::blue), 1.0));
    rollElements.Curve.attach(this);

    pitchElements.Curve.setRenderHint(QwtPlotCurve::RenderAntialiased);
    pitchElements.Curve.setPen(QPen(QColor(Qt::red), 1.0));
    pitchElements.Curve.attach(this);

    yawElements.Curve.setRenderHint(QwtPlotCurve::RenderAntialiased);
    yawElements.Curve.setPen(QPen(QColor(Qt::green), 1.0));
    yawElements.Curve.attach(this);

    rollElements.Curve2.setRenderHint(QwtPlotCurve::RenderAntialiased);
    rollElements.Curve2.setPen(QPen(QColor(Qt::darkBlue), 1.0, Qt::DashLine));
    rollElements.Curve2.setYAxis(QwtPlot::yRight);

    pitchElements.Curve2.setRenderHint(QwtPlotCurve::RenderAntialiased);
    pitchElements.Curve2.setPen(QPen(QColor(Qt::darkRed), 1.0, Qt::DashLine));
    pitchElements.Curve2.setYAxis(QwtPlot::yRight);

    yawElements.Curve2.setRenderHint(QwtPlotCurve::RenderAntialiased);
    yawElements.Curve2.setPen(QPen(QColor(Qt::darkGreen), 1.0, Qt::DashLine));
    yawElements.Curve2.setYAxis(QwtPlot::yRight);

    // legend
    // Show a legend at the top
    QwtLegend *m_legend = new QwtLegend(this);
    m_legend->setDefaultItemMode(QwtLegendData::Checkable);
    m_legend->setFrameStyle(QFrame::Box | QFrame::Sunken);
    m_legend->setToolTip(tr("Click legend to show/hide expo curve"));
    connect(m_legend, SIGNAL(checked(const QVariant &, bool, int)), this, SLOT(showCurve(QVariant,bool,int)));

    QPalette pal = m_legend->palette();
    pal.setColor(m_legend->backgroundRole(), QColor(100, 100, 100));	// background colour
    pal.setColor(QPalette::Text, QColor(0, 0, 0));			// text colour
    m_legend->setPalette(pal);

    insertLegend(m_legend, QwtPlot::TopLegend);

    // axis
    this->enableAxis(QwtPlot::yRight);
    this->setAxisTitle(QwtPlot::xBottom, " normalized stick input");

    STEPS = 1000;
    x_data =  new double[STEPS];
    y_data =  new double[STEPS];

    double step   = 2*1.0 / (STEPS - 1);
    for (int i = 0; i < STEPS; i++) {
        x_data[i] = (i*step) - 1.0;
    }

    HorizonTransition = 0;


    //marker for horizon transition value
    QwtSymbol *sym1 = new QwtSymbol(QwtSymbol::Star1 ,QBrush(Qt::blue),QPen(Qt::blue),QSize(7,7));
    rollElements.Mark.setSymbol(sym1);
    rollElements.Mark_.setSymbol(sym1);
    QwtSymbol *sym1_2 = new QwtSymbol(QwtSymbol::Star1 ,QBrush(Qt::darkBlue),QPen(Qt::darkBlue),QSize(7,7));
    rollElements.Mark2.setSymbol(sym1_2);
    rollElements.Mark2_.setSymbol(sym1_2);
    rollElements.Mark2.setYAxis(QwtPlot::yRight);
    rollElements.Mark2_.setYAxis(QwtPlot::yRight);

    QwtSymbol *sym2 = new QwtSymbol(QwtSymbol::Star1 ,QBrush(Qt::red),QPen(Qt::red),QSize(7,7));
    pitchElements.Mark.setSymbol(sym2);
    pitchElements.Mark_.setSymbol(sym2);
    QwtSymbol *sym2_2 = new QwtSymbol(QwtSymbol::Star1 ,QBrush(Qt::darkRed),QPen(Qt::darkRed),QSize(7,7));
    pitchElements.Mark2.setSymbol(sym2_2);
    pitchElements.Mark2_.setSymbol(sym2_2);
    pitchElements.Mark2.setYAxis(QwtPlot::yRight);
    pitchElements.Mark2_.setYAxis(QwtPlot::yRight);

    QwtSymbol *sym3 = new QwtSymbol(QwtSymbol::Star1 ,QBrush(Qt::green),QPen(Qt::green),QSize(7,7));
    yawElements.Mark.setSymbol(sym3);
    yawElements.Mark_.setSymbol(sym3);
    QwtSymbol *sym3_2 = new QwtSymbol(QwtSymbol::Star1 ,QBrush(Qt::darkGreen),QPen(Qt::darkGreen),QSize(7,7));
    yawElements.Mark2.setSymbol(sym3_2);
    yawElements.Mark2_.setSymbol(sym3_2);
    yawElements.Mark2.setYAxis(QwtPlot::yRight);
    yawElements.Mark2_.setYAxis(QwtPlot::yRight);
}

/**
 * @brief ExpoCurve::init //! Set label for the stick channels
 * @param lbl_mode 0: Label for rate mode, 1: Label for horizon mode
 * @param horizon_transitions 0: no marker, >0: % horizon transitions defined in /flight/Modules/Stabilization/stabilization.c
 * @param roll_value value for initial roll curve
 * @param pitch_value value for initial pitch curve
 * @param yaw_value value for initial yaw curve
 * @param roll_max max for initial roll curve
 * @param pitch_max max for initial pitch curve
 * @param yaw_max max for initial yaw curve
 */
void ExpoCurve::init(int lbl_mode,int horizon_transistion,int roll_value,int pitch_value,int yaw_value,int roll_max,int pitch_max,int yaw_max,int roll_max2,int pitch_max2,int yaw_max2)
{
    switch (lbl_mode)
    {
        case 0:
            rollElements.Curve.setTitle("Roll rate (deg/s)");
            pitchElements.Curve.setTitle("Pitch rate (deg/s)");
            yawElements.Curve.setTitle("Yaw rate (deg/s)");

            this->setAxisTitle(QwtPlot::yLeft, "rate (deg/s)");
            this->setAxisTitle(QwtPlot::yRight, "rate (deg/s)");
            break;
        case 1:
            rollElements.Curve.setTitle("Roll angle (deg)");
            pitchElements.Curve.setTitle("Pitch angle (deg)");
            yawElements.Curve.setTitle("Yaw angle (deg)");
            rollElements.Curve2.setTitle("Roll rate (deg/s)");
            pitchElements.Curve2.setTitle("Pitch rate (deg/s)");
            yawElements.Curve2.setTitle("Yaw rate (deg/s)");
            CurveCnt = 2;
            this->setAxisTitle(QwtPlot::yLeft, "horizon angle (deg)");
            this->setAxisTitle(QwtPlot::yRight, "horizon rate (deg/s)");

            this->setToolTip(tr("This plot shows data only for the Horizon mode, not the Attitude mode.<br><br>For each axis there are 2 curves.<br>One for the 'Horizon Attitude' part (horizon angle) and in darker color one for the 'Horizon Rate' part (horizon rate).<br><br>The markers on the curves show the threshold point:<br><br>- For stick inputs above its purely the 'Horizon Rate' mode with expo scaling by the 'Expo Horizon'<br><br>- For stick inputs below there is a dynamic transistion from 'Horizon Attitude' to 'Horizon Rate' mode."));
            break;
    }

    HorizonTransition = horizon_transistion;

    if (HorizonTransition != 0) {
        rollElements.Mark.attach(this);
        rollElements.Mark_.attach(this);
        pitchElements.Mark.attach(this);
        pitchElements.Mark_.attach(this);
        yawElements.Mark.attach(this);
        yawElements.Mark_.attach(this);

        CurveCnt = 2;

        //this->enableAxis(QwtPlot::yRight);
        rollElements.Curve2.attach(this);
        pitchElements.Curve2.attach(this);
        yawElements.Curve2.attach(this);

        rollElements.Mark2.attach(this);
        rollElements.Mark2_.attach(this);
        pitchElements.Mark2.attach(this);
        pitchElements.Mark2_.attach(this);
        yawElements.Mark2.attach(this);
        yawElements.Mark2_.attach(this);
    }
    else {
        CurveCnt = 1;
    }

    plotDataRoll(roll_value,roll_max,1);
    plotDataPitch(pitch_value,pitch_max,1);
    plotDataYaw(yaw_value,yaw_max,1);

    if (CurveCnt == 2) {
        plotDataRoll(roll_value,roll_max2,2);
        plotDataPitch(pitch_value,pitch_max2,2);
        plotDataYaw(yaw_value,yaw_max2,2);
    }

}

/**
 * @brief ExpoCurve::plotData Show expo data for one of the stick channels
 * @param value The expo coefficient; sets the exponential amount [0,100]
 * @param curve The curve that has to be plot (roll,nick,yaw)
 * @param mark The horizon marker that has to be plot (roll,nick,yaw) if != 0
 *
 * The math here is copied/ the same as in the expo3() function in /flight/Libraries/math/misc_math.c
 * Please be aware of changes that are made there.
 */
void ExpoCurve::plotData(int value, int max, ExpoPlotElements_t &plot_elements, int mode)
{
    double marker_x;
    double marker_y;

    for (int i = 0; i < STEPS; i++) {
        y_data[i] = max*(x_data[i]  * ((100 - value) / 100.0) + pow(x_data[i] , 3) * (value / 100.0));
    }

    if (mode == 1) {
        plot_elements.Curve.setSamples(x_data, y_data, STEPS);
        plot_elements.Curve.show();
    }
    else if (mode == 2) {
        plot_elements.Curve2.setSamples(x_data, y_data, STEPS);
        plot_elements.Curve2.show();
    }

    if (HorizonTransition != 0) {
        marker_x = invers_expo3((double)(HorizonTransition/100.0),value);
        marker_y = max*HorizonTransition/100;

        // additional scaling (*0.985) of positive x marker position for better visual fit with the curve
        if (mode == 1) {
            plot_elements.Mark.setValue(marker_x*0.985,marker_y);
            plot_elements.Mark_.setValue(-1*marker_x,-1*marker_y);
        }
        else if (mode == 2) {
            plot_elements.Mark2.setValue(marker_x*0.985,marker_y);
            plot_elements.Mark2_.setValue(-1*marker_x,-1*marker_y);
        }
    }

    this->replot();

    if(CurveCnt == 1) {
        this->setAxisScaleDiv(yRight, this->axisScaleDiv(yLeft));
        this->replot();
    }

}

void ExpoCurve::plotDataRoll(double value, int max, int mode)
{
    plotData((int)value, max, this->rollElements, mode);
}

void ExpoCurve::plotDataPitch(double value, int max, int mode)
{
    plotData((int)value, max, this->pitchElements, mode);
}

void ExpoCurve::plotDataYaw(double value, int max, int mode)
{
    plotData((int)value, max, this->yawElements, mode);
}

// Inverse expo3 function from /flight
// @param[in] y   expo output data from [-1,1]
// @param[in] g   sets the exponential amount [0,100]
// @return  rescaled expo output to input data
//
// The math: http://www.wolframalpha.com/input/?i=Solve%5By%3D%28%28g%2F100%29*x%5E3%2B%28%28100-g%29%2F100%29*x%29%2Cx%5D
// e.g. y=0.85 expo = 50% : http://www.wolframalpha.com/input/?i=Solve%5B0.85%3D%28%2850%2F100%29*x%5E3%2B%28%28100-50%29%2F100%29*x%29%2Cx%5D
double ExpoCurve::invers_expo3(double y,int g)
{
    double temp1 = pow((2700*pow(g,2)*y + sqrt(7290000*pow(g,4)*pow(y,2)+108*pow((100-g),3)*pow(g,3))),(1.0/3.0));
    double temp2 = pow(2,(1.0/3.0));

    return (temp1/(3*g*temp2) - ((100-g)*temp2)/temp1);
}

/**
 * @brief ExpoCurve::showCurve
 * @param item
 * @param on
 */
void ExpoCurve::showCurve(const QVariant & itemInfo, bool on, int index)
{
    Q_UNUSED(index);
    QwtPlotItem * item = QwtPlot::infoToItem(itemInfo);
    if (item)
        item->setVisible(!on);

    mutex.lock();
    replot();
    mutex.unlock();
}
