/**
 ******************************************************************************
 *
 * @file       configservowidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Servo input/output configuration panel for the config gadget
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
#ifndef CONFIGINPUTWIDGET_H
#define CONFIGINPUTWIDGET_H

#include <QGraphicsView>
#include <QList>
#include <QPointer>
#include <QRadioButton>
#include <QWidget>
#include <QSvgRenderer>
#include <QGraphicsSvgItem>

#include "ui_input.h"
#include "ui_inputchannelform.h"

#include "uavobjectwidgetutils/configtaskwidget.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include <QWidget>
#include <QList>
#include "inputchannelform.h"

#include "accessorydesired.h"
#include "flightstatus.h"
#include "manualcontrolcommand.h"
#include "manualcontrolsettings.h"
#include "receiveractivity.h"

class Ui_InputWidget;

class ConfigInputWidget: public ConfigTaskWidget
{
	Q_OBJECT
public:
        ConfigInputWidget(QWidget *parent = 0);
        ~ConfigInputWidget();
        enum wizardSteps{wizardWelcome,wizardChooseMode,wizardChooseType,wizardIdentifySticks,wizardIdentifyCenter,wizardIdentifyLimits,wizardIdentifyInverted,wizardFinish,wizardNone};
        enum txMode{mode1,mode2};
        enum txMovements{moveLeftVerticalStick,moveRightVerticalStick,moveLeftHorizontalStick,moveRightHorizontalStick,moveAccess0,moveAccess1,moveAccess2,moveFlightMode,centerAll,moveAll,armingSwitch,nothing};
        enum txMovementType{vertical,horizontal,jump,mix};
        enum txType {acro, heli};
        void startInputWizard() { goToWizard(); }

private:
        // This was set through trial and error. Extensive testing
        // will have to be done before changing it. At a minimum,
        // this should include Turnigy9x, FrSky, S.BUS, and Spektrum
        // transmitters.
        static const int DEBOUNCE_THRESHOLD_COUNT = 1;
        // This was set through trial and error. Extensive testing
        // will have to be done before changing it. At a minimum,
        // this should include Turnigy9x, FrSky, S.BUS, and Spektrum
        // transmitters.
        static const int CHANNEL_IDENTIFICATION_WAIT_TIME_MS = 2500;

        bool growing;
        bool reverse[ManualControlSettings::CHANNELNEUTRAL_NUMELEM];
        txMovements currentMovement;
        int movePos;
        void setTxMovement(txMovements movement);
        Ui_InputWidget *m_config;
        wizardSteps wizardStep;
        QList<QPointer<QWidget> > extraWidgets;
        txMode transmitterMode;
        txType transmitterType;
        struct channelsStruct
        {
            bool operator ==(const channelsStruct& rhs) const
            {
                return((group==rhs.group) &&(number==rhs.number));
            }
            int group;
            int number;
        }lastChannel;
        channelsStruct currentChannel;
        QList<channelsStruct> usedChannels;
        bool channelDetected;
        QEventLoop * loop;
        bool skipflag;

        int currentChannelNum;
        QList<int> heliChannelOrder;
        QList<int> acroChannelOrder;

        ManualControlCommand * manualCommandObj;
        ManualControlCommand::DataFields manualCommandData;
        FlightStatus * flightStatusObj;
        FlightStatus::DataFields flightStatusData;
        AccessoryDesired * accessoryDesiredObj0;
        AccessoryDesired * accessoryDesiredObj1;
        AccessoryDesired * accessoryDesiredObj2;
        AccessoryDesired::DataFields accessoryDesiredData0;
        AccessoryDesired::DataFields accessoryDesiredData1;
        AccessoryDesired::DataFields accessoryDesiredData2;
        UAVObject::Metadata manualControlMdata;
        ManualControlSettings * manualSettingsObj;
        ManualControlSettings::DataFields manualSettingsData;
        ManualControlSettings::DataFields previousManualSettingsData;
        ReceiverActivity * receiverActivityObj;
        ReceiverActivity::DataFields receiverActivityData;
        QMap<QString, UAVObject::Metadata> originalMetaData;

        QSvgRenderer *m_renderer;

        // Background: background
        QGraphicsSvgItem *m_txMainBody;
        QGraphicsSvgItem *m_txLeftStick;
        QGraphicsSvgItem *m_txRightStick;
        QGraphicsSvgItem *m_txAccess0;
        QGraphicsSvgItem *m_txAccess1;
        QGraphicsSvgItem *m_txAccess2;
        QGraphicsSvgItem *m_txFlightMode;
        QGraphicsSvgItem *m_txArming;
        QGraphicsSvgItem *m_txBackground;
        QGraphicsSvgItem *m_txArrows;
        QTransform m_txLeftStickOrig;
        QTransform m_txRightStickOrig;
        QTransform m_txAccess0Orig;
        QTransform m_txAccess1Orig;
        QTransform m_txAccess2Orig;
        QTransform m_txFlightModeCOrig;
        QTransform m_txFlightModeLOrig;
        QTransform m_txFlightModeROrig;
        QTransform m_txArmingSwitchOrigL;
        QTransform m_txArmingSwitchOrigR;
        QTransform m_txMainBodyOrig;
        QTransform m_txArrowsOrig;
        QTimer * animate;
        void resetTxControls();
        void setMoveFromCommand(int command);

        void fastMdata();
        void restoreMdata();

        void setChannel(int);
        void nextChannel();
        void prevChannel();

        void wizardSetUpStep(enum wizardSteps);
        void wizardTearDownStep(enum wizardSteps);

        //! Handle to telemetry manager for monitoring for disconnects
        TelemetryManager* telMngr;
        quint8 scaleSwitchChannel(quint8 channelNumber, quint8 switchPositions);

private slots:
        void wzNext();
        void wzBack();
        void wzCancel();
        void goToWizard();
        void disableWizardButton(int);
        void openHelp();
        void identifyControls();
        void identifyLimits();
        void moveTxControls();
        void moveSticks();
        void dimOtherControls(bool value);
        void moveFMSlider();
        void updatePositionSlider();
        void invertControls();
        void simpleCalibration(bool state);
        void updateCalibration();

protected:
        void resizeEvent(QResizeEvent *event);
        virtual void enableControls(bool enable);
};

#endif
