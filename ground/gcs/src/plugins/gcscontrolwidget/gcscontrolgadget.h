/**
 ******************************************************************************
 *
 * @file       GCSControlgadget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup GCSControlGadgetPlugin GCSControl Gadget Plugin
 * @{
 * @brief A gadget to control the UAV, either from the keyboard or a joystick
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

#ifndef GCSControlGADGET_H_
#define GCSControlGADGET_H_

#include <coreplugin/iuavgadget.h>
#include "gcscontrolgadgetconfiguration.h"
#include "gcscontrolwidgetplugin.h"
#include <QTimer>
#include <QTime>
#include <QUdpSocket>
#include <QHostAddress>
#include <gcscontrolplugin/gcscontrol.h>

#if defined(USE_SDL)
#include "sdlgamepad/sdlgamepad.h"
#endif

// UAVOs
#include "manualcontrolcommand.h"

namespace Core {
class IUAVGadget;
}

class GCSControlGadgetWidget;

using namespace Core;

class GCSControlGadget : public Core::IUAVGadget
{
    Q_OBJECT
public:
    GCSControlGadget(QString classId, GCSControlGadgetWidget *widget, QWidget *parent = 0, QObject *plugin=0);
    ~GCSControlGadget();

    QList<int> context() const { return m_context; }
    QWidget *widget() { return (QWidget *) m_widget; }
    QString contextHelpId() const { return QString(); }

    void loadConfiguration(IUAVGadgetConfiguration* config);

private:
    //! Get the handle to the ManualControlCommand object
    ManualControlCommand* getManualControlCommand();

    //! Get the handle to the GCSReceiver object
    GCSControl* getGcsControl();

    double constrain(double value);

    //! Set the GCS Receiver object
    void setGcsReceiver(double leftX, double leftY, double rightX, double rightY);

    QTime joystickTime;
    GCSControlGadgetWidget *m_widget;
    QList<int> m_context;
    UAVObject::Metadata mccInitialData;

    // The channel mappings to the joystick
    int rollChannel;
    int pitchChannel;
    int yawChannel;
    int throttleChannel;

    //! What kind of transmitter layout to use (Mode1 - Mode4)
    int controlsMode;

    //! Send commands to FC via GCS receiver UAVO
    bool gcsReceiverMode;

    //! Whether GCS control is enabled
    bool enableSending;

    buttonSettingsStruct buttonSettings[8];
    double bound(double input);
    double wrap(double input);
    bool channelReverse[8];
    QUdpSocket *control_sock;

signals:
    void sticksChangedRemotely(double leftX, double leftY, double rightX, double rightY);

protected slots:
    void manualControlCommandUpdated(UAVObject *);
    void sticksChangedLocally(double leftX, double leftY, double rightX, double rightY);
    void readUDPCommand();
    void flightModeChanged(ManualControlSettings::FlightModePositionOptions mode);
    //! Enable or disable sending data
    void enableControl(bool enable);

    // signals from joystick
    void gamepads(quint8 count);
#if defined(USE_SDL)
    void buttonState(ButtonNumber number, bool pressed);
    void axesValues(QListInt16 values);
#endif
};


#endif // GCSControlGADGET_H_
