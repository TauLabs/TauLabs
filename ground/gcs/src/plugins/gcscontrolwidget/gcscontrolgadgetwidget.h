/**
 ******************************************************************************
 *
 * @file       GCSControlgadgetwidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup GCSControlGadgetPlugin GCSControl Gadget Plugin
 * @{
 * @brief A place holder gadget plugin 
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

#ifndef GCSControlGADGETWIDGET_H_
#define GCSControlGADGETWIDGET_H_

#include "manualcontrolsettings.h"
#include "manualcontrolcommand.h"
#include <QMap>
#include <QLabel>

#define UDP_PORT 2323

class Ui_GCSControl;

class GCSControlGadgetWidget : public QLabel
{
    Q_OBJECT

public:
    GCSControlGadgetWidget(QWidget *parent = 0);

    void allowGcsControl(bool allow);
    void setGCSControl(bool newState);
    bool getGCSControl(void);
    void setUDPControl(bool newState);
    bool getUDPControl(void);

signals:
    //! Emitted whenever the UI is clicked on to indicate the new stick positions
    void sticksChanged(double leftX, double leftY, double rightX, double rightY);
    void controlEnabled(bool);
    void flightModeChangedLocaly(ManualControlSettings::FlightModePositionOptions);
public slots:
    //! Signals from parent gadget indicating change from the remote system
    void updateSticks(double leftX, double leftY, double rightX, double rightY);

    // signals from children widgets indicating a local change
    void leftStickClicked(double X, double Y);
    void rightStickClicked(double X, double Y);

protected slots:
    void toggleControl(bool checked);
    void selectFlightMode(int state);
    void flightModeChanged(quint8 mode);
    void toggleUDPControl(int state);
    void armedChanged(quint8 armed);

private:
    Ui_GCSControl *m_gcscontrol;
    double leftX,leftY,rightX,rightY;
    QMap<QString,ManualControlSettings::FlightModePositionOptions> flightModesMap;
};

#endif /* GCSControlGADGETWIDGET_H_ */
