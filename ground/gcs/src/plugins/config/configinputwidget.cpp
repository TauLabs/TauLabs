/**
 ******************************************************************************
 *
 * @file       configinputwidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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

#include "configinputwidget.h"

#include "uavtalk/telemetrymanager.h"

#include <QDebug>
#include <QStringList>
#include <QtGui/QWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QPushButton>
#include <QDesktopServices>
#include <QUrl>
#include <QMessageBox>
#include <utils/stylehelper.h>
#include <QMessageBox>

#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>

#define ACCESS_MIN_MOVE -3
#define ACCESS_MAX_MOVE 3
#define STICK_MIN_MOVE -8
#define STICK_MAX_MOVE 8

ConfigInputWidget::ConfigInputWidget(QWidget *parent) : ConfigTaskWidget(parent),wizardStep(wizardNone),transmitterType(heli),loop(NULL),skipflag(false)
{
    controlCommandObj = ControlCommand::GetInstance(getObjectManager());
    controlCommandSettingsObj = ControlCommandSettings::GetInstance(getObjectManager());
    rcTransmitterInputObj = RCTransmitterInput::GetInstance(getObjectManager());
    rcTransmitterSettingsObj = RCTransmitterSettings::GetInstance(getObjectManager());
    flightStatusObj = FlightStatus::GetInstance(getObjectManager());
    receiverActivityObj=ReceiverActivity::GetInstance(getObjectManager());
    m_config = new Ui_InputWidget();
    m_config->setupUi(this);
    
    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if(!settings->useExpertMode())
        m_config->saveRCInputToRAM->setVisible(false);
    
    addApplySaveButtons(m_config->saveRCInputToRAM,m_config->saveRCInputToSD);

    //Generate the rows of buttons in the input channel form GUI
    unsigned int index=0;
    foreach (QString name, rcTransmitterSettingsObj->getField("ChannelNumber")->getElementNames())
    {
        Q_ASSERT(index < RCTransmitterSettings::CHANNELGROUPS_NUMELEM);
        inputChannelForm * inpForm=new inputChannelForm(this,index==0);
        m_config->channelSettings->layout()->addWidget(inpForm); //Add the row to the UI
        inpForm->setName(name);
        addUAVObjectToWidgetRelation("RCTransmitterSettings","ChannelGroups",inpForm->ui->channelGroup,index);
        addUAVObjectToWidgetRelation("RCTransmitterSettings","ChannelNumber",inpForm->ui->channelNumber,index);
        addUAVObjectToWidgetRelation("RCTransmitterSettings","ChannelMin",inpForm->ui->channelMin,index);
        addUAVObjectToWidgetRelation("RCTransmitterSettings","ChannelNeutral",inpForm->ui->channelNeutral,index);
        addUAVObjectToWidgetRelation("RCTransmitterSettings","ChannelMax",inpForm->ui->channelMax,index);
        ++index;
    }

    addUAVObjectToWidgetRelation("RCTransmitterSettings", "Deadband", m_config->deadband, 0, 0.01f);

    connect(m_config->configurationWizard,SIGNAL(clicked()),this,SLOT(goToWizard()));
    connect(m_config->stackedWidget,SIGNAL(currentChanged(int)),this,SLOT(disableWizardButton(int)));
    connect(m_config->runCalibration,SIGNAL(toggled(bool)),this, SLOT(simpleCalibration(bool)));

    connect(m_config->wzNext,SIGNAL(clicked()),this,SLOT(wzNext()));
    connect(m_config->wzCancel,SIGNAL(clicked()),this,SLOT(wzCancel()));
    connect(m_config->wzBack,SIGNAL(clicked()),this,SLOT(wzBack()));

    m_config->stackedWidget->setCurrentIndex(0);
    addUAVObjectToWidgetRelation("ControlCommandSettings","FlightModePosition",m_config->fmsModePos1,0,1,true);
    addUAVObjectToWidgetRelation("ControlCommandSettings","FlightModePosition",m_config->fmsModePos2,1,1,true);
    addUAVObjectToWidgetRelation("ControlCommandSettings","FlightModePosition",m_config->fmsModePos3,2,1,true);
    addUAVObjectToWidgetRelation("ControlCommandSettings","FlightModePosition",m_config->fmsModePos4,3,1,true);
    addUAVObjectToWidgetRelation("ControlCommandSettings","FlightModePosition",m_config->fmsModePos5,4,1,true);
    addUAVObjectToWidgetRelation("ControlCommandSettings","FlightModePosition",m_config->fmsModePos6,5,1,true);
    addUAVObjectToWidgetRelation("ControlCommandSettings","FlightModeNumber",m_config->fmsPosNum);

    addUAVObjectToWidgetRelation("ControlCommandSettings","Stabilization1Settings",m_config->fmsSsPos1Roll,"Roll");
    addUAVObjectToWidgetRelation("ControlCommandSettings","Stabilization2Settings",m_config->fmsSsPos2Roll,"Roll");
    addUAVObjectToWidgetRelation("ControlCommandSettings","Stabilization3Settings",m_config->fmsSsPos3Roll,"Roll");
    addUAVObjectToWidgetRelation("ControlCommandSettings","Stabilization1Settings",m_config->fmsSsPos1Pitch,"Pitch");
    addUAVObjectToWidgetRelation("ControlCommandSettings","Stabilization2Settings",m_config->fmsSsPos2Pitch,"Pitch");
    addUAVObjectToWidgetRelation("ControlCommandSettings","Stabilization3Settings",m_config->fmsSsPos3Pitch,"Pitch");
    addUAVObjectToWidgetRelation("ControlCommandSettings","Stabilization1Settings",m_config->fmsSsPos1Yaw,"Yaw");
    addUAVObjectToWidgetRelation("ControlCommandSettings","Stabilization2Settings",m_config->fmsSsPos2Yaw,"Yaw");
    addUAVObjectToWidgetRelation("ControlCommandSettings","Stabilization3Settings",m_config->fmsSsPos3Yaw,"Yaw");

    addUAVObjectToWidgetRelation("ControlCommandSettings","Arming",m_config->armControl);
    addUAVObjectToWidgetRelation("ControlCommandSettings","ArmedTimeout",m_config->armTimeout,0,1000);
    connect( RCTransmitterInput::GetInstance(getObjectManager()),SIGNAL(objectUpdated(UAVObject*)),this,SLOT(moveFMSlider()));
    connect( ControlCommandSettings::GetInstance(getObjectManager()),SIGNAL(objectUpdated(UAVObject*)),this,SLOT(updatePositionSlider()));
    enableControls(false);

    populateWidgets();
    refreshWidgetsValues();
    // Connect the help button
    connect(m_config->inputHelp, SIGNAL(clicked()), this, SLOT(openHelp()));

    m_config->graphicsView->setScene(new QGraphicsScene(this));
    m_config->graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    m_renderer = new QSvgRenderer();
    QGraphicsScene *l_scene = m_config->graphicsView->scene();
    m_config->graphicsView->setBackgroundBrush(QBrush(Utils::StyleHelper::baseColor()));
    if (QFile::exists(":/configgadget/images/TX2.svg") && m_renderer->load(QString(":/configgadget/images/TX2.svg")) && m_renderer->isValid())
    {
        l_scene->clear(); // Deletes all items contained in the scene as well.

        m_txBackground = new QGraphicsSvgItem();
        // All other items will be clipped to the shape of the background
        m_txBackground->setFlags(QGraphicsItem::ItemClipsChildrenToShape|
                                 QGraphicsItem::ItemClipsToShape);
        m_txBackground->setSharedRenderer(m_renderer);
        m_txBackground->setElementId("background");
        l_scene->addItem(m_txBackground);

        m_txMainBody = new QGraphicsSvgItem();
        m_txMainBody->setParentItem(m_txBackground);
        m_txMainBody->setSharedRenderer(m_renderer);
        m_txMainBody->setElementId("body");

        m_txLeftStick = new QGraphicsSvgItem();
        m_txLeftStick->setParentItem(m_txBackground);
        m_txLeftStick->setSharedRenderer(m_renderer);
        m_txLeftStick->setElementId("ljoy");

        m_txRightStick = new QGraphicsSvgItem();
        m_txRightStick->setParentItem(m_txBackground);
        m_txRightStick->setSharedRenderer(m_renderer);
        m_txRightStick->setElementId("rjoy");

        m_txAccess0 = new QGraphicsSvgItem();
        m_txAccess0->setParentItem(m_txBackground);
        m_txAccess0->setSharedRenderer(m_renderer);
        m_txAccess0->setElementId("access0");

        m_txAccess1 = new QGraphicsSvgItem();
        m_txAccess1->setParentItem(m_txBackground);
        m_txAccess1->setSharedRenderer(m_renderer);
        m_txAccess1->setElementId("access1");

        m_txAccess2 = new QGraphicsSvgItem();
        m_txAccess2->setParentItem(m_txBackground);
        m_txAccess2->setSharedRenderer(m_renderer);
        m_txAccess2->setElementId("access2");

        m_txFlightMode = new QGraphicsSvgItem();
        m_txFlightMode->setParentItem(m_txBackground);
        m_txFlightMode->setSharedRenderer(m_renderer);
        m_txFlightMode->setElementId("flightModeCenter");
        m_txFlightMode->setZValue(-10);

        m_txArrows = new QGraphicsSvgItem();
        m_txArrows->setParentItem(m_txBackground);
        m_txArrows->setSharedRenderer(m_renderer);
        m_txArrows->setElementId("arrows");
        m_txArrows->setVisible(false);

        QRectF orig=m_renderer->boundsOnElement("ljoy");
        QMatrix Matrix = m_renderer->matrixForElement("ljoy");
        orig=Matrix.mapRect(orig);
        m_txLeftStickOrig.translate(orig.x(),orig.y());
        m_txLeftStick->setTransform(m_txLeftStickOrig,false);

        orig=m_renderer->boundsOnElement("arrows");
        Matrix = m_renderer->matrixForElement("arrows");
        orig=Matrix.mapRect(orig);
        m_txArrowsOrig.translate(orig.x(),orig.y());
        m_txArrows->setTransform(m_txArrowsOrig,false);

        orig=m_renderer->boundsOnElement("body");
        Matrix = m_renderer->matrixForElement("body");
        orig=Matrix.mapRect(orig);
        m_txMainBodyOrig.translate(orig.x(),orig.y());
        m_txMainBody->setTransform(m_txMainBodyOrig,false);

        orig=m_renderer->boundsOnElement("flightModeCenter");
        Matrix = m_renderer->matrixForElement("flightModeCenter");
        orig=Matrix.mapRect(orig);
        m_txFlightModeCOrig.translate(orig.x(),orig.y());
        m_txFlightMode->setTransform(m_txFlightModeCOrig,false);

        orig=m_renderer->boundsOnElement("flightModeLeft");
        Matrix = m_renderer->matrixForElement("flightModeLeft");
        orig=Matrix.mapRect(orig);
        m_txFlightModeLOrig.translate(orig.x(),orig.y());
        orig=m_renderer->boundsOnElement("flightModeRight");
        Matrix = m_renderer->matrixForElement("flightModeRight");
        orig=Matrix.mapRect(orig);
        m_txFlightModeROrig.translate(orig.x(),orig.y());

        orig=m_renderer->boundsOnElement("rjoy");
        Matrix = m_renderer->matrixForElement("rjoy");
        orig=Matrix.mapRect(orig);
        m_txRightStickOrig.translate(orig.x(),orig.y());
        m_txRightStick->setTransform(m_txRightStickOrig,false);

        orig=m_renderer->boundsOnElement("access0");
        Matrix = m_renderer->matrixForElement("access0");
        orig=Matrix.mapRect(orig);
        m_txAccess0Orig.translate(orig.x(),orig.y());
        m_txAccess0->setTransform(m_txAccess0Orig,false);

        orig=m_renderer->boundsOnElement("access1");
        Matrix = m_renderer->matrixForElement("access1");
        orig=Matrix.mapRect(orig);
        m_txAccess1Orig.translate(orig.x(),orig.y());
        m_txAccess1->setTransform(m_txAccess1Orig,false);

        orig=m_renderer->boundsOnElement("access2");
        Matrix = m_renderer->matrixForElement("access2");
        orig=Matrix.mapRect(orig);
        m_txAccess2Orig.translate(orig.x(),orig.y());
        m_txAccess2->setTransform(m_txAccess2Orig,true);
    }
    m_config->graphicsView->fitInView(m_txMainBody, Qt::KeepAspectRatio );
    animate=new QTimer(this);
    connect(animate,SIGNAL(timeout()),this,SLOT(moveTxControls()));

    heliChannelOrder << RCTransmitterSettings::CHANNELGROUPS_COLLECTIVE <<
                        RCTransmitterSettings::CHANNELGROUPS_THROTTLE <<
                        RCTransmitterSettings::CHANNELGROUPS_ROLL <<
                        RCTransmitterSettings::CHANNELGROUPS_PITCH <<
                        RCTransmitterSettings::CHANNELGROUPS_YAW <<
                        RCTransmitterSettings::CHANNELGROUPS_FLIGHTMODE <<
                        RCTransmitterSettings::CHANNELGROUPS_ACCESSORY0 <<
                        RCTransmitterSettings::CHANNELGROUPS_ACCESSORY1 <<
                        RCTransmitterSettings::CHANNELGROUPS_ACCESSORY2;

    acroChannelOrder << RCTransmitterSettings::CHANNELGROUPS_THROTTLE <<
                        RCTransmitterSettings::CHANNELGROUPS_ROLL <<
                        RCTransmitterSettings::CHANNELGROUPS_PITCH <<
                        RCTransmitterSettings::CHANNELGROUPS_YAW <<
                        RCTransmitterSettings::CHANNELGROUPS_FLIGHTMODE <<
                        RCTransmitterSettings::CHANNELGROUPS_ACCESSORY0 <<
                        RCTransmitterSettings::CHANNELGROUPS_ACCESSORY1 <<
                        RCTransmitterSettings::CHANNELGROUPS_ACCESSORY2;
}
void ConfigInputWidget::resetTxControls()
{

    m_txLeftStick->setTransform(m_txLeftStickOrig,false);
    m_txRightStick->setTransform(m_txRightStickOrig,false);
    m_txAccess0->setTransform(m_txAccess0Orig,false);
    m_txAccess1->setTransform(m_txAccess1Orig,false);
    m_txAccess2->setTransform(m_txAccess2Orig,false);
    m_txFlightMode->setElementId("flightModeCenter");
    m_txFlightMode->setTransform(m_txFlightModeCOrig,false);
    m_txArrows->setVisible(false);
}

ConfigInputWidget::~ConfigInputWidget()
{

}

void ConfigInputWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    m_config->graphicsView->fitInView(m_txBackground, Qt::KeepAspectRatio );
}

void ConfigInputWidget::openHelp()
{
    QDesktopServices::openUrl( QUrl("http://wiki.openpilot.org/x/04Cf", QUrl::StrictMode) );
}

void ConfigInputWidget::goToWizard()
{
    QMessageBox msgBox;
    msgBox.setText(tr("Arming Settings will be set to Always Disarmed for your safety."));
    msgBox.setDetailedText(tr("You will have to reconfigure the arming settings manually "
                              "when the wizard is finished. After the last step of the "
                              "wizard you will be taken to the Arming Settings screen."));
    msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
    msgBox.setDefaultButton(QMessageBox::Cancel);
    int ret = msgBox.exec();

    switch(ret) {
    case QMessageBox::Ok:
        // Set correct tab visible before starting wizard.
        if(m_config->tabWidget->currentIndex() != 0) {
            m_config->tabWidget->setCurrentIndex(0);
        }
        wizardSetUpStep(wizardWelcome);
        m_config->graphicsView->fitInView(m_txBackground, Qt::KeepAspectRatio );
        break;
    case QMessageBox::Cancel:
        break;
    default:
        break;
    }
}

void ConfigInputWidget::disableWizardButton(int value)
{
    if(value!=0)
        m_config->configurationWizard->setVisible(false);
    else
        m_config->configurationWizard->setVisible(true);
}

void ConfigInputWidget::wzCancel()
{
    dimOtherControls(false);
    m_config->stackedWidget->setCurrentIndex(0);

    if(wizardStep != wizardNone)
        wizardTearDownStep(wizardStep);
    wizardStep=wizardNone;
    m_config->stackedWidget->setCurrentIndex(0);

    // Load manual settings back from beginning of wizard
    controlCommandSettingsObj->setData(previousControlCommandSettingsData);
    rcTransmitterSettingsObj->setData(previousRCTransmitterSettingsData);

    // Load original metadata
    restoreMdata();
}

void ConfigInputWidget::wzNext()
{
    // In identify sticks mode the next button can indicate
    // channel advance
    if(wizardStep != wizardNone &&
            wizardStep != wizardIdentifySticks)
        wizardTearDownStep(wizardStep);

    // State transitions for next button
    switch(wizardStep) {
    case wizardWelcome:
        wizardSetUpStep(wizardChooseMode);
        break;
    case wizardChooseMode:
        wizardSetUpStep(wizardChooseType);
        break;
    case wizardChooseType:
        wizardSetUpStep(wizardIdentifySticks);
        break;
    case wizardIdentifySticks:
        nextChannel();
        if(currentChannelNum==-1) { // Gone through all channels
            wizardTearDownStep(wizardIdentifySticks);
            wizardSetUpStep(wizardIdentifyCenter);
        }
        break;
    case wizardIdentifyCenter:
        wizardSetUpStep(wizardIdentifyLimits);
        break;
    case wizardIdentifyLimits:
        wizardSetUpStep(wizardIdentifyInverted);
        break;
    case wizardIdentifyInverted:
        wizardSetUpStep(wizardFinish);
        break;
    case wizardFinish:
        wizardStep=wizardNone;
        m_config->stackedWidget->setCurrentIndex(0);
        m_config->tabWidget->setCurrentIndex(2);
        break;
    default:
        Q_ASSERT(0);
    }
}

void ConfigInputWidget::wzBack()
{
    if(wizardStep != wizardNone &&
            wizardStep != wizardIdentifySticks)
        wizardTearDownStep(wizardStep);

    // State transitions for next button
    switch(wizardStep) {
    case wizardChooseMode:
        wizardSetUpStep(wizardWelcome);
        break;
    case wizardChooseType:
        wizardSetUpStep(wizardChooseMode);
        break;
    case wizardIdentifySticks:
        prevChannel();
        if(currentChannelNum == -1) {
            wizardTearDownStep(wizardIdentifySticks);
            wizardSetUpStep(wizardChooseType);
        }
        break;
    case wizardIdentifyCenter:
        wizardSetUpStep(wizardIdentifySticks);
        break;
    case wizardIdentifyLimits:
        wizardSetUpStep(wizardIdentifyCenter);
        break;
    case wizardIdentifyInverted:
        wizardSetUpStep(wizardIdentifyLimits);
        break;
    case wizardFinish:
        wizardSetUpStep(wizardIdentifyInverted);
        break;
    default:
        Q_ASSERT(0);
    }
}

void ConfigInputWidget::wizardSetUpStep(enum wizardSteps step)
{
    switch(step) {
    case wizardWelcome:
        foreach(QPointer<QWidget> wd,extraWidgets)
        {
            if(!wd.isNull())
                delete wd;
        }
        extraWidgets.clear();
        m_config->graphicsView->setVisible(false);
        setTxMovement(nothing);

        // Back up settings
        previousRCTransmitterSettingsData = rcTransmitterSettingsObj->getData();
        previousControlCommandSettingsData = controlCommandSettingsObj->getData();

        // Set to always disarmed
        ControlCommandSettings::DataFields controlCommandSettingsData;
        controlCommandSettingsData = controlCommandSettingsObj->getData();
        controlCommandSettingsData.Arming = ControlCommandSettings::ARMING_ALWAYSDISARMED;
        controlCommandSettingsObj->setData(controlCommandSettingsData);

        m_config->wzText->setText(tr("Welcome to the inputs configuration wizard.\n"
                                     "Please follow the instructions on the screen and only move your controls when asked to.\n"
                                     "Make sure you already configured your hardware settings on the proper tab and restarted your board.\n"
                                     "You can press 'back' at any time to return to the previous screeen or press 'Cancel' to quit the wizard.\n"));
        m_config->stackedWidget->setCurrentIndex(1);
        m_config->wzBack->setEnabled(false);
        break;
    case wizardChooseMode:
    {
        m_config->graphicsView->setVisible(true);
        m_config->graphicsView->fitInView(m_txBackground, Qt::KeepAspectRatio );
        setTxMovement(nothing);
        m_config->wzText->setText(tr("Please choose your transmitter type.\n"
                                     "Mode 1 means your throttle stick is on the right.\n"
                                     "Mode 2 means your throttle stick is on the left.\n"));
        m_config->wzBack->setEnabled(true);
        QRadioButton * mode1=new QRadioButton(tr("Mode 1"),this);
        QRadioButton * mode2=new QRadioButton(tr("Mode 2"),this);
        mode2->setChecked(true);
        extraWidgets.clear();
        extraWidgets.append(mode1);
        extraWidgets.append(mode2);
        m_config->checkBoxesLayout->layout()->addWidget(mode1);
        m_config->checkBoxesLayout->layout()->addWidget(mode2);
    }
        break;
    case wizardChooseType:
    {
        m_config->wzText->setText(tr("Please choose your transmitter mode.\n"
                                     "Acro means normal transmitter.\n"
                                     "Heli means there is a collective pitch and throttle input.\n"
                                     "If you are using a heli transmitter please engage throttle hold now.\n"));
        m_config->wzBack->setEnabled(true);
        QRadioButton * typeAcro=new QRadioButton(tr("Acro"),this);
        QRadioButton * typeHeli=new QRadioButton(tr("Heli"),this);
        typeAcro->setChecked(true);
        typeHeli->setChecked(false);
        extraWidgets.clear();
        extraWidgets.append(typeAcro);
        extraWidgets.append(typeHeli);
        m_config->checkBoxesLayout->layout()->addWidget(typeAcro);
        m_config->checkBoxesLayout->layout()->addWidget(typeHeli);
        wizardStep=wizardChooseType;
    }
        break;
    case wizardIdentifySticks:
        usedChannels.clear();
        currentChannelNum=-1;
        nextChannel();
//        manualSettingsData=manualSettingsObj->getData();
        connect(receiverActivityObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(identifyControls()));
        fastMdata();
        m_config->wzNext->setEnabled(false);
        break;
    case wizardIdentifyCenter:
        setTxMovement(centerAll);
        m_config->wzText->setText(QString(tr("Please center all controls and press next when ready (if your FlightMode switch has only two positions, leave it in either position).")));
        fastMdata();
        break;
    case wizardIdentifyLimits:
    {
        accessoryDesiredObj0 = AccessoryDesired::GetInstance(getObjectManager(),0);
        accessoryDesiredObj1 = AccessoryDesired::GetInstance(getObjectManager(),1);
        accessoryDesiredObj2 = AccessoryDesired::GetInstance(getObjectManager(),2);
        setTxMovement(nothing);
        m_config->wzText->setText(QString(tr("Please move all controls to their maximum extents on both directions and press next when ready.")));
        fastMdata();

        RCTransmitterSettings::DataFields rcTransmitterSettingsData;
        rcTransmitterSettingsData = rcTransmitterSettingsObj->getData();
        for(uint i=0;i<RCTransmitterSettings::CHANNELMAX_NUMELEM;++i)
        {
            // Preserve the inverted status
            if(rcTransmitterSettingsData.ChannelMin[i] <= rcTransmitterSettingsData.ChannelMax[i]) {
                rcTransmitterSettingsData.ChannelMin[i] = rcTransmitterSettingsData.ChannelNeutral[i];
                rcTransmitterSettingsData.ChannelMax[i] = rcTransmitterSettingsData.ChannelNeutral[i];
            } else {
                // Make this detect as still inverted
                rcTransmitterSettingsData.ChannelMin[i] = rcTransmitterSettingsData.ChannelNeutral[i] + 1;
                rcTransmitterSettingsData.ChannelMax[i] = rcTransmitterSettingsData.ChannelNeutral[i];
            }
        }
        connect(rcTransmitterInputObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(identifyLimits()));
        connect(controlCommandObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        connect(flightStatusObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        connect(accessoryDesiredObj0, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));

        rcTransmitterSettingsObj->setData(rcTransmitterSettingsData);
    }
        break;
    case wizardIdentifyInverted:
    {
        dimOtherControls(true);
        setTxMovement(nothing);
        extraWidgets.clear();

        RCTransmitterSettings::DataFields rcTransmitterSettingsData;
        rcTransmitterSettingsData = rcTransmitterSettingsObj->getData();

        for (int index = 0; index < rcTransmitterSettingsObj->getField("ChannelMax")->getElementNames().length(); index++)
        {
            QString name = rcTransmitterSettingsObj->getField("ChannelMax")->getElementNames().at(index);
            if(!name.contains("Access") &&  !name.contains("Flight"))
            {
                QCheckBox * cb=new QCheckBox(name,this);
                // Make sure checked status matches current one
                cb->setChecked(rcTransmitterSettingsData.ChannelMax[index] < rcTransmitterSettingsData.ChannelMin[index]);

                extraWidgets.append(cb);
                m_config->checkBoxesLayout->layout()->addWidget(cb);

                connect(cb,SIGNAL(toggled(bool)),this,SLOT(invertControls()));
            }
        }
        connect(controlCommandObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        m_config->wzText->setText(QString(tr("Please check the picture below and correct all the sticks which show an inverted movement, press next when ready.")));
        fastMdata();
    }
        break;
    case wizardFinish:
    {
        dimOtherControls(false);
        connect(controlCommandObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        connect(flightStatusObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        connect(accessoryDesiredObj0, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        m_config->wzText->setText(QString(tr("You have completed this wizard, please check below if the picture mimics your sticks movement.\n"
                                             "These new settings aren't saved to the board yet, after pressing next you will go to the Arming Settings "
                                             "screen where you can set your desired arming sequence and save the configuration.")));
        fastMdata();

        RCTransmitterSettings::DataFields rcTransmitterSettingsData;
        rcTransmitterSettingsData = rcTransmitterSettingsObj->getData();

        rcTransmitterSettingsData.ChannelNeutral[RCTransmitterSettings::CHANNELNEUTRAL_THROTTLE] =
                rcTransmitterSettingsData.ChannelMin[RCTransmitterSettings::CHANNELMIN_THROTTLE] +
                ((rcTransmitterSettingsData.ChannelMax[RCTransmitterSettings::CHANNELMAX_THROTTLE] -
                  rcTransmitterSettingsData.ChannelMin[RCTransmitterSettings::CHANNELMIN_THROTTLE]) * 0.02);
        if((abs(rcTransmitterSettingsData.ChannelMax[RCTransmitterSettings::CHANNELMAX_FLIGHTMODE] - rcTransmitterSettingsData.ChannelNeutral[RCTransmitterSettings::CHANNELNEUTRAL_FLIGHTMODE]) < 100) ||
                (abs(rcTransmitterSettingsData.ChannelMin[RCTransmitterSettings::CHANNELMIN_FLIGHTMODE] - rcTransmitterSettingsData.ChannelNeutral[RCTransmitterSettings::CHANNELNEUTRAL_FLIGHTMODE]) < 100))
        {
            rcTransmitterSettingsData.ChannelNeutral[RCTransmitterSettings::CHANNELNEUTRAL_FLIGHTMODE] = rcTransmitterSettingsData.ChannelMin[RCTransmitterSettings::CHANNELMIN_FLIGHTMODE] +
                    (rcTransmitterSettingsData.ChannelMax[RCTransmitterSettings::CHANNELMAX_FLIGHTMODE] - rcTransmitterSettingsData.ChannelMin[RCTransmitterSettings::CHANNELMIN_FLIGHTMODE]) / 2;
        }
        rcTransmitterSettingsObj->setData(rcTransmitterSettingsData);
    }
        break;
    default:
        Q_ASSERT(0);
    }
    wizardStep = step;
}

void ConfigInputWidget::wizardTearDownStep(enum wizardSteps step)
{
    QRadioButton * mode, * type;
    Q_ASSERT(step == wizardStep);
    switch(step) {
    case wizardWelcome:
        break;
    case wizardChooseMode:
        mode=qobject_cast<QRadioButton *>(extraWidgets.at(0));
        if(mode->isChecked())
            transmitterMode=mode1;
        else
            transmitterMode=mode2;
        delete extraWidgets.at(0);
        delete extraWidgets.at(1);
        extraWidgets.clear();
        break;
    case wizardChooseType:
        type=qobject_cast<QRadioButton *>(extraWidgets.at(0));
        if(type->isChecked())
            transmitterType=acro;
        else
            transmitterType=heli;
        delete extraWidgets.at(0);
        delete extraWidgets.at(1);
        extraWidgets.clear();
        break;
    case wizardIdentifySticks:
        disconnect(receiverActivityObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(identifyControls()));
        m_config->wzNext->setEnabled(true);
        setTxMovement(nothing);
        break;
    case wizardIdentifyCenter:
        rcTransmitterInputData = rcTransmitterInputObj->getData();
        RCTransmitterSettings::DataFields rcTransmitterSettingsData;
        rcTransmitterSettingsData = rcTransmitterSettingsObj->getData();

        for(unsigned int i=0;i<RCTransmitterInput::CHANNEL_NUMELEM;++i)
        {
            rcTransmitterSettingsData.ChannelNeutral[i]=rcTransmitterInputData.Channel[i];
        }
        rcTransmitterSettingsObj->setData(rcTransmitterSettingsData);

        setTxMovement(nothing);
        break;
    case wizardIdentifyLimits:
        disconnect(rcTransmitterInputObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(identifyLimits()));
        disconnect(controlCommandObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        disconnect(flightStatusObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        disconnect(accessoryDesiredObj0, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
//        manualSettingsObj->setData(manualSettingsData);
        setTxMovement(nothing);
        break;
    case wizardIdentifyInverted:
        dimOtherControls(false);
        foreach(QWidget * wd,extraWidgets)
        {
            QCheckBox * cb=qobject_cast<QCheckBox *>(wd);
            if(cb)
            {
                disconnect(cb,SIGNAL(toggled(bool)),this,SLOT(invertControls()));
                delete cb;
            }
        }
        extraWidgets.clear();
        disconnect(controlCommandObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        break;
    case wizardFinish:
        dimOtherControls(false);
        setTxMovement(nothing);
        disconnect(controlCommandObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        disconnect(flightStatusObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        disconnect(accessoryDesiredObj0, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(moveSticks()));
        restoreMdata();
        break;
    default:
        Q_ASSERT(0);
    }
}

/**
 * @brief ConfigInputWidget::fastMdata Set manual control command to fast updates. Set all others to updates slowly.
 */
void ConfigInputWidget::fastMdata()
{
    // Check that the metadata hasn't already been saved.
    if (!originalMetaData.empty())
        return;

    // Store original metadata
    UAVObjectUtilManager* utilMngr = getObjectUtilManager();
    originalMetaData = utilMngr->readAllNonSettingsMetadata();

    // Update data rates
    uint16_t slowUpdate = 5000; // in [ms]
    uint16_t fastUpdate =  150; // in [ms]

    // Iterate over list of UAVObjects, configuring all dynamic data metadata objects.
    UAVObjectManager *objManager = getObjectManager();
    QMap<QString, UAVObject::Metadata> metaDataList;
    QVector< QVector<UAVDataObject*> > objList = objManager->getDataObjects();
    foreach (QVector<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            if(!obj->isSettings()) {
                UAVObject::Metadata mdata = obj->getMetadata();
                UAVObject::SetFlightAccess(mdata, UAVObject::ACCESS_READWRITE);

                switch(obj->getObjID()){
                case ReceiverActivity::OBJID:
                    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
                    break;
                case AccessoryDesired::OBJID:
                case FlightStatus::OBJID:
                case ControlCommand::OBJID:
                    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
                    mdata.flightTelemetryUpdatePeriod = fastUpdate;
                    break;
                case RCTransmitterInput::OBJID:
                    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
                    mdata.flightTelemetryUpdatePeriod = fastUpdate;
                    break;
                default:
                    UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_PERIODIC);
                    mdata.flightTelemetryUpdatePeriod = slowUpdate;
                }

                metaDataList.insert(obj->getName(), mdata);

            }
        }
    }

    // Set new metadata
    utilMngr->setAllNonSettingsMetadata(metaDataList);


}

/**
  * Restore previous metadata settings
  */
void ConfigInputWidget::restoreMdata()
{
    UAVObjectUtilManager* utilMngr = getObjectUtilManager();
    utilMngr->setAllNonSettingsMetadata(originalMetaData);
    originalMetaData.clear();
}

/**
  * Set the display to indicate which channel the person should move
  */
void ConfigInputWidget::setChannel(int newChan)
{
    if(newChan == RCTransmitterSettings::CHANNELGROUPS_COLLECTIVE)
        m_config->wzText->setText(QString(tr("Please enable the throttle hold mode and move the collective pitch stick.")));
    else if (newChan == RCTransmitterSettings::CHANNELGROUPS_FLIGHTMODE)
        m_config->wzText->setText(QString(tr("Please toggle the flight mode switch. For switches you may have to repeat this rapidly.")));
    else if((transmitterType == heli) && (newChan == RCTransmitterSettings::CHANNELGROUPS_THROTTLE))
        m_config->wzText->setText(QString(tr("Please disable throttle hold mode and move the throttle stick.")));
    else
        m_config->wzText->setText(QString(tr("Please move each control once at a time according to the instructions and picture below.\n\n"
                                 "Move the %1 stick")).arg(rcTransmitterSettingsObj->getField("ChannelGroups")->getElementNames().at(newChan)));

    if(rcTransmitterSettingsObj->getField("ChannelGroups")->getElementNames().at(newChan).contains("Accessory") ||
       rcTransmitterSettingsObj->getField("ChannelGroups")->getElementNames().at(newChan).contains("FlightMode")) {
        m_config->wzNext->setEnabled(true);
        m_config->wzText->setText(m_config->wzText->text() + tr(" or click next to skip this channel."));
    } else
        m_config->wzNext->setEnabled(false);

    setMoveFromCommand(newChan);

    currentChannelNum = newChan;
    channelDetected = false;
}

/**
  * Unfortunately order of channel should be different in different conditions.  Selects
  * next channel based on heli or acro mode
  */
void ConfigInputWidget::nextChannel()
{
    QList <int> order = (transmitterType == heli) ? heliChannelOrder : acroChannelOrder;

    if(currentChannelNum == -1) {
        setChannel(order[0]);
        return;
    }
    for (int i = 0; i < order.length() - 1; i++) {
        if(order[i] == currentChannelNum) {
            setChannel(order[i+1]);
            return;
        }
    }
    currentChannelNum = -1; // hit end of list
}

/**
  * Unfortunately order of channel should be different in different conditions.  Selects
  * previous channel based on heli or acro mode
  */
void ConfigInputWidget::prevChannel()
{
    QList <int> order = transmitterType == heli ? heliChannelOrder : acroChannelOrder;

    // No previous from unset channel or next state
    if(currentChannelNum == -1)
        return;

    for (int i = 1; i < order.length(); i++) {
        if(order[i] == currentChannelNum) {
            setChannel(order[i-1]);
            usedChannels.removeLast();
            return;
        }
    }
    currentChannelNum = -1; // hit end of list
}


/**
 * @brief ConfigInputWidget::identifyControls Triggers whenever a new ReceiverActivity UAVO is received.
 * This function checks if the channel has already been identified, and if not assigns it to a new
 * RCTransmitterSettings channel.
 */
void ConfigInputWidget::identifyControls()
{
    static int debounce=0;

    receiverActivityData=receiverActivityObj->getData();
    if(receiverActivityData.ActiveChannel==255)
        return;

    if(channelDetected)
        return;
    else
    {
        receiverActivityData=receiverActivityObj->getData();
        currentChannel.group=receiverActivityData.ActiveGroup;
        currentChannel.number=receiverActivityData.ActiveChannel;
        if(currentChannel==lastChannel)
            ++debounce;
        lastChannel.group= currentChannel.group;
        lastChannel.number=currentChannel.number;

        // If this channel isn't already allocated, and the debounce passes the threshold, set
        // the input channel as detected.
        if(!usedChannels.contains(lastChannel) && debounce>DEBOUNCE_THRESHOLD_COUNT)
        {
            channelDetected = true;
            debounce=0;
            usedChannels.append(lastChannel);
            RCTransmitterSettings::DataFields rcTransmitterSettingsData;
            rcTransmitterSettingsData = rcTransmitterSettingsObj->getData();
            rcTransmitterSettingsData.ChannelGroups[currentChannelNum] = currentChannel.group;
            rcTransmitterSettingsData.ChannelNumber[currentChannelNum] = currentChannel.number;
            rcTransmitterSettingsObj->setData(rcTransmitterSettingsData);
        }
        else
            return;
    }

    // At this point in the code, the channel has been successfully identified
    m_config->wzText->clear();
    setTxMovement(nothing);

    // Wait to ensure that the user is no longer moving the sticks. Empricially,
    // this delay works well across a broad range of users and transmitters.
    QTimer::singleShot(CHANNEL_IDENTIFICATION_WAIT_TIME_MS, this, SLOT(wzNext()));
}

void ConfigInputWidget::identifyLimits()
{
    rcTransmitterInputData=rcTransmitterInputObj->getData();
    RCTransmitterSettings::DataFields rcTransmitterSettingsData;
    rcTransmitterSettingsData = rcTransmitterSettingsObj->getData();

    for(uint i=0; i<RCTransmitterSettings::CHANNELMAX_NUMELEM; ++i)
    {
        if(rcTransmitterSettingsData.ChannelMin[i] <= rcTransmitterSettingsData.ChannelMax[i]) {
            // Non inverted channel
            if(rcTransmitterSettingsData.ChannelMin[i]>rcTransmitterInputData.Channel[i])
                rcTransmitterSettingsData.ChannelMin[i]=rcTransmitterInputData.Channel[i];
            if(rcTransmitterSettingsData.ChannelMax[i]<rcTransmitterInputData.Channel[i])
                rcTransmitterSettingsData.ChannelMax[i]=rcTransmitterInputData.Channel[i];
        } else {
            // Inverted channel
            if(rcTransmitterSettingsData.ChannelMax[i]>rcTransmitterInputData.Channel[i])
                rcTransmitterSettingsData.ChannelMax[i]=rcTransmitterInputData.Channel[i];
            if(rcTransmitterSettingsData.ChannelMin[i]<rcTransmitterInputData.Channel[i])
                rcTransmitterSettingsData.ChannelMin[i]=rcTransmitterInputData.Channel[i];
        }
    }
    rcTransmitterSettingsObj->setData(rcTransmitterSettingsData);
}
void ConfigInputWidget::setMoveFromCommand(int command)
{
    //CHANNELNUMBER_ROLL=0, CHANNELNUMBER_PITCH=1, CHANNELNUMBER_YAW=2, CHANNELNUMBER_THROTTLE=3, CHANNELNUMBER_FLIGHTMODE=4, CHANNELNUMBER_ACCESSORY0=5, CHANNELNUMBER_ACCESSORY1=6, CHANNELNUMBER_ACCESSORY2=7 } ChannelNumberElem;
    if(command==RCTransmitterSettings::CHANNELNUMBER_ROLL)
    {
            setTxMovement(moveRightHorizontalStick);
    }
    else if(command==RCTransmitterSettings::CHANNELNUMBER_PITCH)
    {
        if(transmitterMode==mode2)
            setTxMovement(moveRightVerticalStick);
        else
            setTxMovement(moveLeftVerticalStick);
    }
    else if(command==RCTransmitterSettings::CHANNELNUMBER_YAW)
    {
            setTxMovement(moveLeftHorizontalStick);
    }
    else if(command==RCTransmitterSettings::CHANNELNUMBER_THROTTLE)
    {
        if(transmitterMode==mode2)
            setTxMovement(moveLeftVerticalStick);
        else
            setTxMovement(moveRightVerticalStick);
    }
    else if(command==RCTransmitterSettings::CHANNELNUMBER_COLLECTIVE)
    {
        if(transmitterMode==mode2)
            setTxMovement(moveLeftVerticalStick);
        else
            setTxMovement(moveRightVerticalStick);
    }
    else if(command==RCTransmitterSettings::CHANNELNUMBER_FLIGHTMODE)
    {
        setTxMovement(moveFlightMode);
    }
    else if(command==RCTransmitterSettings::CHANNELNUMBER_ACCESSORY0)
    {
        setTxMovement(moveAccess0);
    }
    else if(command==RCTransmitterSettings::CHANNELNUMBER_ACCESSORY1)
    {
        setTxMovement(moveAccess1);
    }
    else if(command==RCTransmitterSettings::CHANNELNUMBER_ACCESSORY2)
    {
        setTxMovement(moveAccess2);
    }

}

void ConfigInputWidget::setTxMovement(txMovements movement)
{
    resetTxControls();
    switch(movement)
    {
    case moveLeftVerticalStick:
        movePos=0;
        growing=true;
        currentMovement=moveLeftVerticalStick;
        animate->start(100);
        break;
    case moveRightVerticalStick:
        movePos=0;
        growing=true;
        currentMovement=moveRightVerticalStick;
        animate->start(100);
        break;
    case moveLeftHorizontalStick:
        movePos=0;
        growing=true;
        currentMovement=moveLeftHorizontalStick;
        animate->start(100);
        break;
    case moveRightHorizontalStick:
        movePos=0;
        growing=true;
        currentMovement=moveRightHorizontalStick;
        animate->start(100);
        break;
    case moveAccess0:
        movePos=0;
        growing=true;
        currentMovement=moveAccess0;
        animate->start(100);
        break;
    case moveAccess1:
        movePos=0;
        growing=true;
        currentMovement=moveAccess1;
        animate->start(100);
        break;
    case moveAccess2:
        movePos=0;
        growing=true;
        currentMovement=moveAccess2;
        animate->start(100);
        break;
    case moveFlightMode:
        movePos=0;
        growing=true;
        currentMovement=moveFlightMode;
        animate->start(1000);
        break;
    case centerAll:
        movePos=0;
        currentMovement=centerAll;
        animate->start(1000);
        break;
    case moveAll:
        movePos=0;
        growing=true;
        currentMovement=moveAll;
        animate->start(50);
        break;
    case nothing:
        movePos=0;
        animate->stop();
        break;
    default:
        break;
    }
}

void ConfigInputWidget::moveTxControls()
{
    QTransform trans;
    QGraphicsItem * item;
    txMovementType move;
    int limitMax;
    int limitMin;
    static bool auxFlag=false;
    switch(currentMovement)
    {
    case moveLeftVerticalStick:
        item=m_txLeftStick;
        trans=m_txLeftStickOrig;
        limitMax=STICK_MAX_MOVE;
        limitMin=STICK_MIN_MOVE;
        move=vertical;
        break;
    case moveRightVerticalStick:
        item=m_txRightStick;
        trans=m_txRightStickOrig;
        limitMax=STICK_MAX_MOVE;
        limitMin=STICK_MIN_MOVE;
        move=vertical;
        break;
    case moveLeftHorizontalStick:
        item=m_txLeftStick;
        trans=m_txLeftStickOrig;
        limitMax=STICK_MAX_MOVE;
        limitMin=STICK_MIN_MOVE;
        move=horizontal;
        break;
    case moveRightHorizontalStick:
        item=m_txRightStick;
        trans=m_txRightStickOrig;
        limitMax=STICK_MAX_MOVE;
        limitMin=STICK_MIN_MOVE;
        move=horizontal;
        break;
    case moveAccess0:
        item=m_txAccess0;
        trans=m_txAccess0Orig;
        limitMax=ACCESS_MAX_MOVE;
        limitMin=ACCESS_MIN_MOVE;
        move=horizontal;
        break;
    case moveAccess1:
        item=m_txAccess1;
        trans=m_txAccess1Orig;
        limitMax=ACCESS_MAX_MOVE;
        limitMin=ACCESS_MIN_MOVE;
        move=horizontal;
        break;
    case moveAccess2:
        item=m_txAccess2;
        trans=m_txAccess2Orig;
        limitMax=ACCESS_MAX_MOVE;
        limitMin=ACCESS_MIN_MOVE;
        move=horizontal;
        break;
    case moveFlightMode:
        item=m_txFlightMode;
        move=jump;
        break;
    case centerAll:
        item=m_txArrows;
        move=jump;
        break;
    case moveAll:
        limitMax=STICK_MAX_MOVE;
        limitMin=STICK_MIN_MOVE;
        move=mix;
        break;
    default:
        break;
    }
    if(move==vertical)
        item->setTransform(trans.translate(0,movePos*10),false);
    else if(move==horizontal)
        item->setTransform(trans.translate(movePos*10,0),false);
    else if(move==jump)
    {
        if(item==m_txArrows)
        {
            m_txArrows->setVisible(!m_txArrows->isVisible());
        }
        else if(item==m_txFlightMode)
        {
            QGraphicsSvgItem * svg;
            svg=(QGraphicsSvgItem *)item;
            if (svg)
            {
                if(svg->elementId()=="flightModeCenter")
                {
                    if(growing)
                    {
                        svg->setElementId("flightModeRight");
                        m_txFlightMode->setTransform(m_txFlightModeROrig,false);
                    }
                    else
                    {
                        svg->setElementId("flightModeLeft");
                        m_txFlightMode->setTransform(m_txFlightModeLOrig,false);
                    }
                }
                else if(svg->elementId()=="flightModeRight")
                {
                    growing=false;
                    svg->setElementId("flightModeCenter");
                    m_txFlightMode->setTransform(m_txFlightModeCOrig,false);
                }
                else if(svg->elementId()=="flightModeLeft")
                {
                    growing=true;
                    svg->setElementId("flightModeCenter");
                    m_txFlightMode->setTransform(m_txFlightModeCOrig,false);
                }
            }
        }
    }
    else if(move==mix)
    {
        trans=m_txAccess0Orig;
        m_txAccess0->setTransform(trans.translate(movePos*10*ACCESS_MAX_MOVE/STICK_MAX_MOVE,0),false);
        trans=m_txAccess1Orig;
        m_txAccess1->setTransform(trans.translate(movePos*10*ACCESS_MAX_MOVE/STICK_MAX_MOVE,0),false);
        trans=m_txAccess2Orig;
        m_txAccess2->setTransform(trans.translate(movePos*10*ACCESS_MAX_MOVE/STICK_MAX_MOVE,0),false);

        if(auxFlag)
        {
            trans=m_txLeftStickOrig;
            m_txLeftStick->setTransform(trans.translate(0,movePos*10),false);
            trans=m_txRightStickOrig;
            m_txRightStick->setTransform(trans.translate(0,movePos*10),false);
        }
        else
        {
            trans=m_txLeftStickOrig;
            m_txLeftStick->setTransform(trans.translate(movePos*10,0),false);
            trans=m_txRightStickOrig;
            m_txRightStick->setTransform(trans.translate(movePos*10,0),false);
        }

        if(movePos==0)
        {
            m_txFlightMode->setElementId("flightModeCenter");
            m_txFlightMode->setTransform(m_txFlightModeCOrig,false);
        }
        else if(movePos==ACCESS_MAX_MOVE/2)
        {
            m_txFlightMode->setElementId("flightModeRight");
            m_txFlightMode->setTransform(m_txFlightModeROrig,false);
        }
        else if(movePos==ACCESS_MIN_MOVE/2)
        {
            m_txFlightMode->setElementId("flightModeLeft");
            m_txFlightMode->setTransform(m_txFlightModeLOrig,false);
        }
    }
    if(move==horizontal || move==vertical ||move==mix)
    {
        if(movePos==0 && growing)
            auxFlag=!auxFlag;
        if(growing)
            ++movePos;
        else
            --movePos;
        if(movePos>limitMax)
        {
            movePos=movePos-2;
            growing=false;
        }
        if(movePos<limitMin)
        {
            movePos=movePos+2;
            growing=true;
        }
    }
}

void ConfigInputWidget::moveSticks()
{
    QTransform trans;
    ControlCommandSettings::DataFields controlCommandSettingsData;
    controlCommandSettingsData = controlCommandSettingsObj->getData();
    controlCommandData=controlCommandObj->getData();
    flightStatusData=flightStatusObj->getData();
    accessoryDesiredData0=accessoryDesiredObj0->getData();
    accessoryDesiredData1=accessoryDesiredObj1->getData();
    accessoryDesiredData2=accessoryDesiredObj2->getData();

    if(transmitterMode==mode2)
    {
        trans=m_txLeftStickOrig;
        m_txLeftStick->setTransform(trans.translate(controlCommandData.Yaw*STICK_MAX_MOVE*10,-controlCommandData.Throttle*STICK_MAX_MOVE*10),false);
        trans=m_txRightStickOrig;
        m_txRightStick->setTransform(trans.translate(controlCommandData.Roll*STICK_MAX_MOVE*10,controlCommandData.Pitch*STICK_MAX_MOVE*10),false);
    }
    else
    {
        trans=m_txRightStickOrig;
        m_txRightStick->setTransform(trans.translate(controlCommandData.Roll*STICK_MAX_MOVE*10, -controlCommandData.Throttle*STICK_MAX_MOVE*10), false);
        trans=m_txLeftStickOrig;
        m_txLeftStick->setTransform(trans.translate(controlCommandData.Yaw*STICK_MAX_MOVE*10, controlCommandData.Pitch*STICK_MAX_MOVE*10), false);
    }
    if(flightStatusData.FlightMode==controlCommandSettingsData.FlightModePosition[0])
    {
        m_txFlightMode->setElementId("flightModeLeft");
        m_txFlightMode->setTransform(m_txFlightModeLOrig,false);
    }
    else if (flightStatusData.FlightMode==controlCommandSettingsData.FlightModePosition[1])
    {
        m_txFlightMode->setElementId("flightModeCenter");
        m_txFlightMode->setTransform(m_txFlightModeCOrig,false);
    }
    else if (flightStatusData.FlightMode==controlCommandSettingsData.FlightModePosition[2])
    {
        m_txFlightMode->setElementId("flightModeRight");
        m_txFlightMode->setTransform(m_txFlightModeROrig,false);
    }
    m_txAccess0->setTransform(QTransform(m_txAccess0Orig).translate(accessoryDesiredData0.AccessoryVal*ACCESS_MAX_MOVE*10,0),false);
    m_txAccess1->setTransform(QTransform(m_txAccess1Orig).translate(accessoryDesiredData1.AccessoryVal*ACCESS_MAX_MOVE*10,0),false);
    m_txAccess2->setTransform(QTransform(m_txAccess2Orig).translate(accessoryDesiredData2.AccessoryVal*ACCESS_MAX_MOVE*10,0),false);
}

void ConfigInputWidget::dimOtherControls(bool value)
{
    qreal opac;
    if(value)
        opac=0.1;
    else
        opac=1;
    m_txAccess0->setOpacity(opac);
    m_txAccess1->setOpacity(opac);
    m_txAccess2->setOpacity(opac);
    m_txFlightMode->setOpacity(opac);
}

void ConfigInputWidget::enableControls(bool enable)
{
    m_config->configurationWizard->setEnabled(enable);
    m_config->runCalibration->setEnabled(enable);

    ConfigTaskWidget::enableControls(enable);

}

void ConfigInputWidget::invertControls()
{
    RCTransmitterSettings::DataFields rcTransmitterSettingsData;
    rcTransmitterSettingsData = rcTransmitterSettingsObj->getData();
    foreach(QWidget * wd,extraWidgets)
    {
        QCheckBox * cb=qobject_cast<QCheckBox *>(wd);
        if(cb)
        {
            int index = rcTransmitterSettingsObj->getField("ChannelNumber")->getElementNames().indexOf(cb->text());
            if((cb->isChecked() && (rcTransmitterSettingsData.ChannelMax[index]>rcTransmitterSettingsData.ChannelMin[index])) ||
                    (!cb->isChecked() && (rcTransmitterSettingsData.ChannelMax[index]<rcTransmitterSettingsData.ChannelMin[index])))
            {
                qint16 aux;
                aux=rcTransmitterSettingsData.ChannelMax[index];
                rcTransmitterSettingsData.ChannelMax[index]=rcTransmitterSettingsData.ChannelMin[index];
                rcTransmitterSettingsData.ChannelMin[index]=aux;
            }
        }
    }
    rcTransmitterSettingsObj->setData(rcTransmitterSettingsData);
}

void ConfigInputWidget::moveFMSlider()
{
    RCTransmitterSettings::DataFields rcTransmitterSettingsDataPriv = rcTransmitterSettingsObj->getData();
    ControlCommandSettings::DataFields controlCommandSettingsDataPriv = controlCommandSettingsObj->getData();
    RCTransmitterInput::DataFields rcTransmitterInputDataPriv = rcTransmitterInputObj->getData();

    float valueScaled;
    int chMin = rcTransmitterSettingsDataPriv.ChannelMin[RCTransmitterSettings::CHANNELMIN_FLIGHTMODE];
    int chMax = rcTransmitterSettingsDataPriv.ChannelMax[RCTransmitterSettings::CHANNELMAX_FLIGHTMODE];
    int chNeutral = rcTransmitterSettingsDataPriv.ChannelNeutral[RCTransmitterSettings::CHANNELNEUTRAL_FLIGHTMODE];

    int value = rcTransmitterInputDataPriv.Channel[RCTransmitterSettings::CHANNELMIN_FLIGHTMODE]; // TODO: Examine this one, as it looks odd to mix the command and settings channels
    if ((chMax > chMin && value >= chNeutral) || (chMin > chMax && value <= chNeutral))
    {
        if (chMax != chNeutral)
            valueScaled = (float)(value - chNeutral) / (float)(chMax - chNeutral);
        else
            valueScaled = 0;
    }
    else
    {
        if (chMin != chNeutral)
            valueScaled = (float)(value - chNeutral) / (float)(chNeutral - chMin);
        else
            valueScaled = 0;
    }

    // Bound and scale FlightMode from [-1..+1] to [0..1] range
    if (valueScaled < -1.0)
        valueScaled = -1.0;
    else
    if (valueScaled >  1.0)
        valueScaled =  1.0;

    // Convert flightMode value into the switch position in the range [0..N-1]
    // This uses the same optimized computation as flight code to be consistent
    uint8_t pos = ((int16_t)(valueScaled * 256) + 256) * controlCommandSettingsDataPriv.FlightModeNumber >> 9;
    if (pos >= controlCommandSettingsDataPriv.FlightModeNumber)
        pos = controlCommandSettingsDataPriv.FlightModeNumber - 1;
    m_config->fmsSlider->setValue(pos);
}

void ConfigInputWidget::updatePositionSlider()
{
    ControlCommandSettings::DataFields controlCommandSettingsDataPriv = controlCommandSettingsObj->getData();

    switch(controlCommandSettingsDataPriv.FlightModeNumber) {
    default:
    case 6:
        m_config->fmsModePos6->setEnabled(true);
	// pass through
    case 5:
        m_config->fmsModePos5->setEnabled(true);
	// pass through
    case 4:
        m_config->fmsModePos4->setEnabled(true);
	// pass through
    case 3:
        m_config->fmsModePos3->setEnabled(true);
	// pass through
    case 2:
        m_config->fmsModePos2->setEnabled(true);
	// pass through
    case 1:
        m_config->fmsModePos1->setEnabled(true);
	// pass through
    case 0:
	break;
    }

    switch(controlCommandSettingsDataPriv.FlightModeNumber) {
    case 0:
        m_config->fmsModePos1->setEnabled(false);
	// pass through
    case 1:
        m_config->fmsModePos2->setEnabled(false);
	// pass through
    case 2:
        m_config->fmsModePos3->setEnabled(false);
	// pass through
    case 3:
        m_config->fmsModePos4->setEnabled(false);
	// pass through
    case 4:
        m_config->fmsModePos5->setEnabled(false);
	// pass through
    case 5:
        m_config->fmsModePos6->setEnabled(false);
	// pass through
    case 6:
    default:
	break;
    }
}

void ConfigInputWidget::updateCalibration()
{
    RCTransmitterSettings::DataFields rcTransmitterSettingsData = rcTransmitterSettingsObj->getData();
    for(uint i=0;i<RCTransmitterSettings::CHANNELMAX_NUMELEM;++i)
    {
        if((!reverse[i] && rcTransmitterSettingsData.ChannelMin[i]>rcTransmitterInputData.Channel[i]) ||
                (reverse[i] && rcTransmitterSettingsData.ChannelMin[i]<rcTransmitterInputData.Channel[i]))
            rcTransmitterSettingsData.ChannelMin[i]=rcTransmitterInputData.Channel[i];
        if((!reverse[i] && rcTransmitterSettingsData.ChannelMax[i]<rcTransmitterInputData.Channel[i]) ||
                (reverse[i] && rcTransmitterSettingsData.ChannelMax[i]>rcTransmitterInputData.Channel[i]))
            rcTransmitterSettingsData.ChannelMax[i]=rcTransmitterInputData.Channel[i];
        rcTransmitterSettingsData.ChannelNeutral[i] = rcTransmitterInputData.Channel[i];
    }

    rcTransmitterSettingsObj->setData(rcTransmitterSettingsData);
    rcTransmitterSettingsObj->updated();
}

void ConfigInputWidget::simpleCalibration(bool enable)
{
    if (enable) {
        m_config->configurationWizard->setEnabled(false);

        QMessageBox msgBox;
        msgBox.setText(tr("Arming Settings are now set to Always Disarmed for your safety."));
        msgBox.setDetailedText(tr("You will have to reconfigure the arming settings manually when the wizard is finished."));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.setDefaultButton(QMessageBox::Ok);
        msgBox.exec();

        rcTransmitterInputData = rcTransmitterInputObj->getData();

        // Set to always disarmed
        ControlCommandSettings::DataFields controlCommandSettingsData;
        controlCommandSettingsData = controlCommandSettingsObj->getData();
        controlCommandSettingsData.Arming = ControlCommandSettings::ARMING_ALWAYSDISARMED;
        controlCommandSettingsObj->setData(controlCommandSettingsData);

        RCTransmitterSettings::DataFields rcTransmitterSettingsData;
        rcTransmitterSettingsData = rcTransmitterSettingsObj->getData();
        for (unsigned int i = 0; i < RCTransmitterInput::CHANNEL_NUMELEM; i++) {
            reverse[i] = rcTransmitterSettingsData.ChannelMax[i] < rcTransmitterSettingsData.ChannelMin[i];
            rcTransmitterSettingsData.ChannelMin[i] = rcTransmitterInputData.Channel[i];
            rcTransmitterSettingsData.ChannelNeutral[i] = rcTransmitterInputData.Channel[i];
            rcTransmitterSettingsData.ChannelMax[i] = rcTransmitterInputData.Channel[i];
        }
        rcTransmitterSettingsObj->setData(rcTransmitterSettingsData);

        fastMdata();

        connect(rcTransmitterSettingsObj, SIGNAL(objectUnpacked(UAVObject*)), this, SLOT(updateCalibration()));
    } else {
        m_config->configurationWizard->setEnabled(true);

        rcTransmitterInputData = rcTransmitterInputObj->getData();
        RCTransmitterSettings::DataFields rcTransmitterSettingsData;
        rcTransmitterSettingsData = rcTransmitterSettingsObj->getData();

        restoreMdata();

        for (unsigned int i = 0; i < RCTransmitterInput::CHANNEL_NUMELEM; i++)
            rcTransmitterSettingsData.ChannelNeutral[i] = rcTransmitterInputData.Channel[i];

        // Force flight mode neutral to middle
        rcTransmitterSettingsData.ChannelNeutral[RCTransmitterSettings::CHANNELNUMBER_FLIGHTMODE] =
                (rcTransmitterSettingsData.ChannelMax[RCTransmitterSettings::CHANNELNUMBER_FLIGHTMODE] +
                rcTransmitterSettingsData.ChannelMin[RCTransmitterSettings::CHANNELNUMBER_FLIGHTMODE]) / 2;

        // Force throttle to be near min
        rcTransmitterSettingsData.ChannelNeutral[RCTransmitterSettings::CHANNELNEUTRAL_THROTTLE]=
                rcTransmitterSettingsData.ChannelMin[RCTransmitterSettings::CHANNELMIN_THROTTLE]+
                ((rcTransmitterSettingsData.ChannelMax[RCTransmitterSettings::CHANNELMAX_THROTTLE]-
                  rcTransmitterSettingsData.ChannelMin[RCTransmitterSettings::CHANNELMIN_THROTTLE])*0.02);

        rcTransmitterSettingsObj->setData(rcTransmitterSettingsData);

        disconnect(rcTransmitterSettingsObj, SIGNAL(objectUnpacked(UAVObject*)), this, SLOT(updateCalibration()));
    }
}
