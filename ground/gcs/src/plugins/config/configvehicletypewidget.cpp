/**
 ******************************************************************************
 *
 * @file       configvehicletypewidget.cpp
 * @author     E. Lafargue, K. Sebesta & The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014 
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Airframe configuration panel
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
#include "configvehicletypewidget.h"

#include <QDebug>
#include <QStringList>
#include <QTimer>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <math.h>
#include <QDesktopServices>
#include <QUrl>
#include <QEventLoop>
#include <QMessageBox>

#include "systemsettings.h"
#include "mixersettings.h"
#include "actuatorsettings.h"
#include "vehicletrim.h"
#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>


/**
  Helper delegate for the custom mixer editor table.
  Taken straight from Qt examples, thanks!
  */
SpinBoxDelegate::SpinBoxDelegate(QObject *parent)
     : QItemDelegate(parent)
 {
 }

QWidget *SpinBoxDelegate::createEditor(QWidget *parent,
    const QStyleOptionViewItem &/* option */,
    const QModelIndex &/* index */) const
{
    QSpinBox *editor = new QSpinBox(parent);
    editor->setMinimum(-127);
    editor->setMaximum(127);

    return editor;
}

void SpinBoxDelegate::setEditorData(QWidget *editor,
                                    const QModelIndex &index) const
{
    int value = index.model()->data(index, Qt::EditRole).toInt();

    QSpinBox *spinBox = static_cast<QSpinBox*>(editor);
    spinBox->setValue(value);
}

void SpinBoxDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                   const QModelIndex &index) const
{
    QSpinBox *spinBox = static_cast<QSpinBox*>(editor);
    spinBox->interpretText();
    int value = spinBox->value();

    model->setData(index, value, Qt::EditRole);
}

void SpinBoxDelegate::updateEditorGeometry(QWidget *editor,
    const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
{
    editor->setGeometry(option.rect);
}

/**********************************************************************************/


/**
 Constructor
 */
ConfigVehicleTypeWidget::ConfigVehicleTypeWidget(QWidget *parent) : ConfigTaskWidget(parent)
{
    m_aircraft = new Ui_AircraftWidget();
    m_aircraft->setupUi(this);
    
    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if(!settings->useExpertMode())
        m_aircraft->saveAircraftToRAM->setVisible(false);

    addApplySaveButtons(m_aircraft->saveAircraftToRAM,m_aircraft->saveAircraftToSD);

    addUAVObject("SystemSettings");
    addUAVObject("MixerSettings");

    addUAVObjectToWidgetRelation("MixerSettings","Curve2Source",m_aircraft->customThrottle2Curve->getCBCurveSource());

    //Generate lists of mixerTypeNames, mixerVectorNames, channelNames
    channelNames << "None";
    for (int i = 0; i < (int)ActuatorSettings::CHANNELTYPE_NUMELEM; i++) {

        mixerTypes << QString("Mixer%1Type").arg(i+1);
        mixerVectors << QString("Mixer%1Vector").arg(i+1);
        channelNames << QString("Channel%1").arg(i+1);
    }

    // Set up vehicle type combobox
    m_aircraft->aircraftType->addItem("Fixed Wing", AIRFRAME_FIXED_WING);
    m_aircraft->aircraftType->addItem("Multirotor", AIRFRAME_MULTIROTOR);
    m_aircraft->aircraftType->addItem("Helicopter", AIRFRAME_HELICOPTER);
    m_aircraft->aircraftType->addItem("Ground", AIRFRAME_GROUND);
    m_aircraft->aircraftType->addItem("Custom", AIRFRAME_CUSTOM);

    m_aircraft->aircraftType->setCurrentIndex(1);    //Set default vehicle to MultiRotor
    m_aircraft->airframesWidget->setCurrentIndex(1); // Force the tab index to match

    // Setup fixed-wing combobox
    m_aircraft->fixedWingType->addItem("Elevator aileron rudder", SystemSettings::AIRFRAMETYPE_FIXEDWING);
    m_aircraft->fixedWingType->addItem("Elevon", SystemSettings::AIRFRAMETYPE_FIXEDWINGELEVON);
    m_aircraft->fixedWingType->addItem("Vtail", SystemSettings::AIRFRAMETYPE_FIXEDWINGVTAIL);
    m_aircraft->fixedWingType->setCurrentIndex(0); //Set default model to "Elevator aileron rudder"

    // Setup ground vehicle combobox
    m_aircraft->groundVehicleType->addItem("Turnable (car)", SystemSettings::AIRFRAMETYPE_GROUNDVEHICLECAR);
    m_aircraft->groundVehicleType->addItem("Differential (tank)", SystemSettings::AIRFRAMETYPE_GROUNDVEHICLEDIFFERENTIAL);
    m_aircraft->groundVehicleType->addItem("Motorcycle", SystemSettings::AIRFRAMETYPE_GROUNDVEHICLEMOTORCYCLE);
    m_aircraft->groundVehicleType->setCurrentIndex(0); //Set default model to "Turnable (car)"

    // Setup multirotor combobox
    m_aircraft->multirotorFrameType->addItem("Tricopter Y", SystemSettings::AIRFRAMETYPE_TRI);
    m_aircraft->multirotorFrameType->addItem("Quad X", SystemSettings::AIRFRAMETYPE_QUADX);
    m_aircraft->multirotorFrameType->addItem("Quad +", SystemSettings::AIRFRAMETYPE_QUADP);
    m_aircraft->multirotorFrameType->addItem("Hexacopter", SystemSettings::AIRFRAMETYPE_HEXA);
    m_aircraft->multirotorFrameType->addItem("Hexacopter X", SystemSettings::AIRFRAMETYPE_HEXAX);
    m_aircraft->multirotorFrameType->addItem("Hexacopter Y6", SystemSettings::AIRFRAMETYPE_HEXACOAX);
    m_aircraft->multirotorFrameType->addItem("Octocopter", SystemSettings::AIRFRAMETYPE_OCTO);
    m_aircraft->multirotorFrameType->addItem("Octocopter V", SystemSettings::AIRFRAMETYPE_OCTOV);
    m_aircraft->multirotorFrameType->addItem("Octocopter Coax +", SystemSettings::AIRFRAMETYPE_OCTOCOAXP);
    m_aircraft->multirotorFrameType->addItem("Octocopter Coax X", SystemSettings::AIRFRAMETYPE_OCTOCOAXX);
    m_aircraft->multirotorFrameType->setCurrentIndex(2); //Set default model to "Quad X"


	//NEW STYLE: Loop through the widgets looking for all widgets that have "ChannelBox" in their name
	//  The upshot of this is that ALL new ComboBox widgets for selecting the output channel must have "ChannelBox" in their name
	foreach(QComboBox *combobox, this->findChildren<QComboBox*>(QRegExp("\\S+ChannelBo\\S+")))//FOR WHATEVER REASON, THIS DOES NOT WORK WITH ChannelBox. ChannelBo is sufficiently accurate
	{
        combobox->addItems(channelNames);
    }
	
    // Setup the Multirotor picture in the Quad settings interface
    m_aircraft->quadShape->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_aircraft->quadShape->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    QSvgRenderer *renderer = new QSvgRenderer();
    renderer->load(QString(":/configgadget/images/multirotor-shapes.svg"));
    quad = new QGraphicsSvgItem();
    quad->setSharedRenderer(renderer);
    quad->setElementId("quad-x");
    QGraphicsScene *scene = new QGraphicsScene(this);
    scene->addItem(quad);
    scene->setSceneRect(quad->boundingRect());
    m_aircraft->quadShape->setScene(scene);

    // Put combo boxes in line one of the custom mixer table:
    UAVDataObject* obj = dynamic_cast<UAVDataObject*>(getObjectManager()->getObject(QString("MixerSettings")));
    UAVObjectField* field = obj->getField(QString("Mixer1Type"));
    QStringList list = field->getOptions();
    for (int i=0; i<(int)(VehicleConfig::CHANNEL_NUMELEM); i++) {
        QComboBox* qb = new QComboBox(m_aircraft->customMixerTable);
        qb->addItems(list);
        m_aircraft->customMixerTable->setCellWidget(0,i,qb);
    }

    SpinBoxDelegate *sbd = new SpinBoxDelegate();
    for (int i=1; i<(int)(VehicleConfig::CHANNEL_NUMELEM); i++) {
        m_aircraft->customMixerTable->setItemDelegateForRow(i, sbd);
    }

    // create and setup a MultiRotor config widget
    m_multirotor = new ConfigMultiRotorWidget(m_aircraft);
    m_multirotor->quad = quad;
    m_multirotor->uiowner = this;
    m_multirotor->setupUI(SystemSettings::AIRFRAMETYPE_QUADX);

    // create and setup a GroundVehicle config widget
    m_groundvehicle = new ConfigGroundVehicleWidget(m_aircraft);
    m_groundvehicle->setupUI(SystemSettings::AIRFRAMETYPE_GROUNDVEHICLECAR);

    // create and setup a FixedWing config widget
    m_fixedwing = new ConfigFixedWingWidget(m_aircraft);
    m_fixedwing->setupUI(SystemSettings::AIRFRAMETYPE_FIXEDWING);

    // create and setup a Helicopter config widget
    m_heli = m_aircraft->helicopterLayout;
    m_heli->setupUI(SystemSettings::AIRFRAMETYPE_HELICP);

	//Connect aircraft type selection dropbox to callback function
    connect(m_aircraft->aircraftType, SIGNAL(currentIndexChanged(int)), this, SLOT(switchAirframeType(int)));
	
	//Connect airframe selection dropbox to callback functions
    connect(m_aircraft->fixedWingType, SIGNAL(currentIndexChanged(int)), this, SLOT(doSetupAirframeUI(int)));
    connect(m_aircraft->multirotorFrameType, SIGNAL(currentIndexChanged(int)), this, SLOT(doSetupAirframeUI(int)));
    connect(m_aircraft->groundVehicleType, SIGNAL(currentIndexChanged(int)), this, SLOT(doSetupAirframeUI(int)));
    //mdl connect(m_heli->m_ccpm->ccpmType, SIGNAL(currentIndexChanged(QString)), this, SLOT(setupAirframeUI(QString)));

    //Connect the multirotor motor reverse checkbox
    connect(m_aircraft->MultirotorRevMixercheckBox, SIGNAL(clicked(bool)), this, SLOT(reverseMultirotorMotor()));

    // Connect actuator and level bias buttons to slots
    connect(m_aircraft->bnLevelTrim, SIGNAL(clicked()), this, SLOT(bnLevelTrim_clicked()));
    connect(m_aircraft->bnServoTrim, SIGNAL(clicked()), this, SLOT(bnServoTrim_clicked()));

    // Connect the help pushbutton
    connect(m_aircraft->airframeHelp, SIGNAL(clicked()), this, SLOT(openHelp()));
    enableControls(false);
    refreshWidgetsValues();
    addToDirtyMonitor();

    disableMouseWheelEvents();
    m_aircraft->quadShape->fitInView(quad, Qt::KeepAspectRatio);
}


/**
 Destructor
 */
ConfigVehicleTypeWidget::~ConfigVehicleTypeWidget()
{
   // Do nothing
}

/**
  Static function to get currently assigned channelDescriptions
  for all known vehicle types;  instantiates the appropriate object
  then asks it to supply channel descs
  */
QStringList ConfigVehicleTypeWidget::getChannelDescriptions()
{    
    int i;
    QStringList channelDesc;

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();
    Q_ASSERT(objMngr);

    // get an instance of systemsettings
    SystemSettings * systemSettings = SystemSettings::GetInstance(objMngr);
    Q_ASSERT(systemSettings);
    SystemSettings::DataFields systemSettingsData = systemSettings->getData();

    switch (systemSettingsData.AirframeType)
    {
        // fixed wing
        case SystemSettings::AIRFRAMETYPE_FIXEDWING:
        case SystemSettings::AIRFRAMETYPE_FIXEDWINGELEVON:
        case SystemSettings::AIRFRAMETYPE_FIXEDWINGVTAIL:
        {
            channelDesc = ConfigFixedWingWidget::getChannelDescriptions();
        }
        break;

        // helicp
        case SystemSettings::AIRFRAMETYPE_HELICP:
        {
            channelDesc = ConfigCcpmWidget::getChannelDescriptions();
        }
        break;

        //multirotor
        case SystemSettings::AIRFRAMETYPE_VTOL:
        case SystemSettings::AIRFRAMETYPE_TRI:
        case SystemSettings::AIRFRAMETYPE_QUADX:
        case SystemSettings::AIRFRAMETYPE_QUADP:
        case SystemSettings::AIRFRAMETYPE_OCTOV:
        case SystemSettings::AIRFRAMETYPE_OCTOCOAXX:
        case SystemSettings::AIRFRAMETYPE_OCTOCOAXP:
        case SystemSettings::AIRFRAMETYPE_OCTO:
        case SystemSettings::AIRFRAMETYPE_HEXAX:
        case SystemSettings::AIRFRAMETYPE_HEXACOAX:
        case SystemSettings::AIRFRAMETYPE_HEXA:
        {
            channelDesc = ConfigMultiRotorWidget::getChannelDescriptions();
        }
        break;

        // ground
        case SystemSettings::AIRFRAMETYPE_GROUNDVEHICLECAR:
        case SystemSettings::AIRFRAMETYPE_GROUNDVEHICLEDIFFERENTIAL:
        case SystemSettings::AIRFRAMETYPE_GROUNDVEHICLEMOTORCYCLE:
        {
            channelDesc = ConfigGroundVehicleWidget::getChannelDescriptions();
        }
        break;

        default:
        {
            for (i=0; i < (int)(VehicleConfig::CHANNEL_NUMELEM); i++)
                channelDesc.append(QString("-"));
        }
        break;
    }

    return channelDesc;
}


/**
  Slot for switching the airframe type. We do it explicitely
  rather than a signal in the UI, because we want to force a fitInView of the quad shapes.
  This is because this method (fitinview) only works when the widget is shown.
  */
void ConfigVehicleTypeWidget::switchAirframeType(int index)
{
    m_aircraft->airframesWidget->setCurrentIndex(index);
    m_aircraft->quadShape->setSceneRect(quad->boundingRect());
    m_aircraft->quadShape->fitInView(quad, Qt::KeepAspectRatio);
    switch(m_aircraft->aircraftType->itemData(index).toInt()) {
    case AIRFRAME_FIXED_WING:
        m_aircraft->bnServoTrim->setEnabled(true);
        break;
    case AIRFRAME_CUSTOM:
        m_aircraft->customMixerTable->resizeColumnsToContents();
        for (int i=0;i<(int)(VehicleConfig::CHANNEL_NUMELEM);i++) {
            m_aircraft->customMixerTable->setColumnWidth(i,(m_aircraft->customMixerTable->width()-
                                                            m_aircraft->customMixerTable->verticalHeader()->width())/10);
        }
    case AIRFRAME_MULTIROTOR:
    case AIRFRAME_HELICOPTER:
    case AIRFRAME_GROUND:
        m_aircraft->bnServoTrim->setEnabled(false);
        break;
    }
}


/**
 WHAT DOES THIS DO???
 */
void ConfigVehicleTypeWidget::showEvent(QShowEvent *event)
{
    Q_UNUSED(event)
    // Thit fitInView method should only be called now, once the
    // widget is shown, otherwise it cannot compute its values and
    // the result is usually a ahrsbargraph that is way too small.
    m_aircraft->quadShape->fitInView(quad, Qt::KeepAspectRatio);
    m_aircraft->customMixerTable->resizeColumnsToContents();
    for (int i=0;i<(int)(VehicleConfig::CHANNEL_NUMELEM);i++) {
        m_aircraft->customMixerTable->setColumnWidth(i,(m_aircraft->customMixerTable->width()-
                                                        m_aircraft->customMixerTable->verticalHeader()->width())/ 10);
    }
}

/**
 Resize the GUI contents when the user changes the window size
 */
void ConfigVehicleTypeWidget::resizeEvent(QResizeEvent* event)
{
    Q_UNUSED(event);
    m_aircraft->quadShape->fitInView(quad, Qt::KeepAspectRatio);
    // Make the custom table columns autostretch:
    m_aircraft->customMixerTable->resizeColumnsToContents();
    for (int i=0;i<(int)(VehicleConfig::CHANNEL_NUMELEM);i++) {
        m_aircraft->customMixerTable->setColumnWidth(i,(m_aircraft->customMixerTable->width()-
                                                        m_aircraft->customMixerTable->verticalHeader()->width())/ 10);
    }

}


void ConfigVehicleTypeWidget::toggleAileron2(int index)
{
    if (index) {
        m_aircraft->fwAileron2ChannelBox->setEnabled(true);
        m_aircraft->fwAileron2Label->setEnabled(true);
    } else {
        m_aircraft->fwAileron2ChannelBox->setEnabled(false);
        m_aircraft->fwAileron2Label->setEnabled(false);
    }
}

void ConfigVehicleTypeWidget::toggleElevator2(int index)
{
    if (index) {
        m_aircraft->fwElevator2ChannelBox->setEnabled(true);
        m_aircraft->fwElevator2Label->setEnabled(true);
    } else {
        m_aircraft->fwElevator2ChannelBox->setEnabled(false);
        m_aircraft->fwElevator2Label->setEnabled(false);
    }
}

void ConfigVehicleTypeWidget::toggleRudder2(int index)
{
    if (index) {
        m_aircraft->fwRudder2ChannelBox->setEnabled(true);
        m_aircraft->fwRudder2Label->setEnabled(true);
    } else {
        m_aircraft->fwRudder2ChannelBox->setEnabled(false);
        m_aircraft->fwRudder2Label->setEnabled(false);
    }
}

/**************************
  * Aircraft settings
  **************************/
/**
  Refreshes the current value of the SystemSettings which holds the aircraft type
  */
void ConfigVehicleTypeWidget::refreshWidgetsValues(UAVObject * obj)
{

    ConfigTaskWidget::refreshWidgetsValues(obj);

    if(!allObjectsUpdated())
        return;
	
    bool dirty=isDirty();
	
    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);

    SystemSettings *systemSettings = SystemSettings::GetInstance(getObjectManager());
    Q_ASSERT(systemSettings);
    SystemSettings::DataFields systemSettingsData = systemSettings->getData();

    // Get the Airframe type from the system settings:
    // At this stage, we will need to have some hardcoded settings in this code, this
    // is not ideal, but there you go.
    frameType = (SystemSettings::AirframeTypeOptions) systemSettingsData.AirframeType;
    setupAirframeUI(frameType);

    QPointer<VehicleConfig> vconfig = new VehicleConfig();

    QList<double> curveValues;    
    vconfig->getThrottleCurve(mixerSettings, MixerSettings::MIXER1VECTOR_THROTTLECURVE1, &curveValues);

    // is at least one of the curve values != 0?
    if (vconfig->isValidThrottleCurve(&curveValues)) {
        // yes, use the curve we just read from mixersettings
        m_aircraft->multiThrottleCurve->initCurve(&curveValues);
        m_aircraft->fixedWingThrottle->initCurve(&curveValues);
        m_aircraft->groundVehicleThrottle1->initCurve(&curveValues);
    }
    else {
        // no, init a straight curve
        m_aircraft->multiThrottleCurve->initLinearCurve(curveValues.count(), 0.9);
        m_aircraft->fixedWingThrottle->initLinearCurve(curveValues.count(), 1.0);
        m_aircraft->groundVehicleThrottle1->initLinearCurve(curveValues.count(), 1.0);
    }
	
    // Setup all Throttle2 curves for all types of airframes //AT THIS MOMENT, THAT MEANS ONLY GROUND VEHICLES
    vconfig->getThrottleCurve(mixerSettings, MixerSettings::MIXER1VECTOR_THROTTLECURVE2, &curveValues);

    if (vconfig->isValidThrottleCurve(&curveValues)) {
        m_aircraft->groundVehicleThrottle2->initCurve(&curveValues);
    }
    else {
        m_aircraft->groundVehicleThrottle2->initLinearCurve(curveValues.count(), 1.0);
    }

    // Load the Settings for vehicle frames:
    switch(frameType) {
    case SystemSettings::AIRFRAMETYPE_FIXEDWING:
    case SystemSettings::AIRFRAMETYPE_FIXEDWINGELEVON:
    case SystemSettings::AIRFRAMETYPE_FIXEDWINGVTAIL:
        // Retrieve fixed wing settings
        m_fixedwing->refreshAirframeWidgetsValues(frameType);
        break;
    case SystemSettings::AIRFRAMETYPE_TRI:
    case SystemSettings::AIRFRAMETYPE_QUADX:
    case SystemSettings::AIRFRAMETYPE_QUADP:
    case SystemSettings::AIRFRAMETYPE_HEXA:
    case SystemSettings::AIRFRAMETYPE_HEXAX:
    case SystemSettings::AIRFRAMETYPE_HEXACOAX:
    case SystemSettings::AIRFRAMETYPE_OCTO:
    case SystemSettings::AIRFRAMETYPE_OCTOV:
    case SystemSettings::AIRFRAMETYPE_OCTOCOAXP:
    case SystemSettings::AIRFRAMETYPE_OCTOCOAXX:
        // Retrieve multirotor settings
        m_multirotor->refreshAirframeWidgetsValues(frameType);
        break;
    case SystemSettings::AIRFRAMETYPE_HELICP:
        // Retrieve helicopter settings
        setComboCurrentIndex(m_aircraft->aircraftType, m_aircraft->aircraftType->findText("Helicopter"));
        m_heli->refreshAirframeWidgetsValues(frameType);
        break;
    case SystemSettings::AIRFRAMETYPE_GROUNDVEHICLECAR:
    case SystemSettings::AIRFRAMETYPE_GROUNDVEHICLEDIFFERENTIAL:
    case SystemSettings::AIRFRAMETYPE_GROUNDVEHICLEMOTORCYCLE:
        // Retrieve ground vehicle settings
        m_groundvehicle->refreshAirframeWidgetsValues(frameType);
        break;
    default:
        // Retrieve custom settings
        setComboCurrentIndex(m_aircraft->aircraftType, m_aircraft->aircraftType->findText("Custom"));
        break;
    }

    updateCustomAirframeUI();

    // Tell UI is has unsaved changes
    setDirty(dirty);
}

/**
 * @brief ConfigVehicleTypeWidget::doSetupAirframeUI Reads combobox value and then calls airframe UI setup method
 * @param comboboxIndex
 */
void ConfigVehicleTypeWidget::doSetupAirframeUI(int comboboxIndex)
{
    // Check which tab page is currently selected, and get the item data from the appropriate combobox
    if (m_aircraft->aircraftType->currentText() == "Multirotor"){
        frameType = (SystemSettings::AirframeTypeOptions) m_aircraft->multirotorFrameType->itemData(comboboxIndex).toUInt();
    }
    else if (m_aircraft->aircraftType->currentText() == "Fixed Wing"){
        frameType = (SystemSettings::AirframeTypeOptions) m_aircraft->fixedWingType->itemData(comboboxIndex).toUInt();
    }
    else if (m_aircraft->aircraftType->currentText() == "Helicopter"){
        frameType = SystemSettings::AIRFRAMETYPE_HELICP;
    }
    else if (m_aircraft->aircraftType->currentText() == "Ground"){
        frameType = (SystemSettings::AirframeTypeOptions) m_aircraft->groundVehicleType->itemData(comboboxIndex).toUInt();
    }
    else if (m_aircraft->aircraftType->currentText() == "Custom"){
        frameType = SystemSettings::AIRFRAMETYPE_CUSTOM;
    }

    // Setup the
    setupAirframeUI(frameType);
}


/**
 * @brief ConfigVehicleTypeWidget::setupAirframeUI Sets up the mixer depending on Airframe type.
 * @param frameType
 */
void ConfigVehicleTypeWidget::setupAirframeUI(SystemSettings::AirframeTypeOptions frameType)
{
    bool dirty=isDirty();

    switch(frameType) {
    case SystemSettings::AIRFRAMETYPE_FIXEDWING:
    case SystemSettings::AIRFRAMETYPE_FIXEDWINGELEVON:
    case SystemSettings::AIRFRAMETYPE_FIXEDWINGVTAIL:
        //Call fixed-wing setup UI
        m_fixedwing->setupUI(frameType);
        break;
    case SystemSettings::AIRFRAMETYPE_TRI:
    case SystemSettings::AIRFRAMETYPE_QUADX:
    case SystemSettings::AIRFRAMETYPE_QUADP:
    case SystemSettings::AIRFRAMETYPE_HEXA:
    case SystemSettings::AIRFRAMETYPE_HEXAX:
    case SystemSettings::AIRFRAMETYPE_HEXACOAX:
    case SystemSettings::AIRFRAMETYPE_OCTO:
    case SystemSettings::AIRFRAMETYPE_OCTOV:
    case SystemSettings::AIRFRAMETYPE_OCTOCOAXP:
    case SystemSettings::AIRFRAMETYPE_OCTOCOAXX:
        //Call multi-rotor setup UI
        m_multirotor->setupUI(frameType);
        break;
    case SystemSettings::AIRFRAMETYPE_HELICP:
        //Call helicopter setup UI
        m_heli->setupUI(frameType);
        break;
    case SystemSettings::AIRFRAMETYPE_GROUNDVEHICLECAR:
    case SystemSettings::AIRFRAMETYPE_GROUNDVEHICLEDIFFERENTIAL:
    case SystemSettings::AIRFRAMETYPE_GROUNDVEHICLEMOTORCYCLE:
        //Call ground vehicle setup UI
        m_groundvehicle->setupUI(frameType);
        break;
    default:
        break;
    }
	
	//SHOULDN'T THIS BE DONE ONLY IN QUAD SETUP, AND NOT ALL THE REST???
    m_aircraft->quadShape->setSceneRect(quad->boundingRect());
    m_aircraft->quadShape->fitInView(quad, Qt::KeepAspectRatio);

    setDirty(dirty);
}


/**
  Reset the contents of a field
  */
void ConfigVehicleTypeWidget::resetField(UAVObjectField * field)
{
    for (unsigned int i=0;i<field->getNumElements();i++) {
        field->setValue(0,i);
    }
}

/**
  Updates the custom airframe settings based on the current airframe.

  Note: does NOT ask for an object refresh itself!
  */
void ConfigVehicleTypeWidget::updateCustomAirframeUI()
{    
    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);

    QPointer<VehicleConfig> vconfig = new VehicleConfig();

    QList<double> curveValues;
    vconfig->getThrottleCurve(mixerSettings, MixerSettings::MIXER1VECTOR_THROTTLECURVE1, &curveValues);

    // is at least one of the curve values != 0?
    if (vconfig->isValidThrottleCurve(&curveValues)) {
        m_aircraft->customThrottle1Curve->initCurve(&curveValues);
    }
    else {
        // no, init a straight curve
        m_aircraft->customThrottle1Curve->initLinearCurve(curveValues.count(), 1.0);
    }

    MixerSettings::DataFields mixerSettingsData = mixerSettings->getData();
    if (mixerSettingsData.Curve2Source == MixerSettings::CURVE2SOURCE_THROTTLE)
        m_aircraft->customThrottle2Curve->setMixerType(MixerCurve::MIXERCURVE_THROTTLE, false);
    else {
        m_aircraft->customThrottle2Curve->setMixerType(MixerCurve::MIXERCURVE_OTHER, false);
    }

    // Setup all Throttle2 curves for all types of airframes
    vconfig->getThrottleCurve(mixerSettings, MixerSettings::MIXER1VECTOR_THROTTLECURVE2, &curveValues);

    if (vconfig->isValidThrottleCurve(&curveValues)) {
        m_aircraft->customThrottle2Curve->initCurve(&curveValues);
    }
    else {
        m_aircraft->customThrottle2Curve->initLinearCurve(curveValues.count(), 1.0, m_aircraft->customThrottle2Curve->getMin());
    }

    // Update the mixer table:
    for (int channel=0; channel<(int)(VehicleConfig::CHANNEL_NUMELEM); channel++) {
        UAVObjectField* field = mixerSettings->getField(mixerTypes.at(channel));
        if (field)
        {
            QComboBox* q = (QComboBox*)m_aircraft->customMixerTable->cellWidget(0,channel);
            if (q)
            {
                QString s = field->getValue().toString();
                setComboCurrentIndex(q, q->findText(s));
            }

            m_aircraft->customMixerTable->item(1,channel)->setText(
                QString::number(vconfig->getMixerVectorValue(mixerSettings,channel,MixerSettings::MIXER1VECTOR_THROTTLECURVE1)));
            m_aircraft->customMixerTable->item(2,channel)->setText(
                QString::number(vconfig->getMixerVectorValue(mixerSettings,channel,MixerSettings::MIXER1VECTOR_THROTTLECURVE2)));
            m_aircraft->customMixerTable->item(3,channel)->setText(
                QString::number(vconfig->getMixerVectorValue(mixerSettings,channel,MixerSettings::MIXER1VECTOR_ROLL)));
            m_aircraft->customMixerTable->item(4,channel)->setText(
                QString::number(vconfig->getMixerVectorValue(mixerSettings,channel,MixerSettings::MIXER1VECTOR_PITCH)));
            m_aircraft->customMixerTable->item(5,channel)->setText(
                QString::number(vconfig->getMixerVectorValue(mixerSettings,channel,MixerSettings::MIXER1VECTOR_YAW)));
        }
    }
}


/**
  Sends the config to the board (airframe type)

  We do all the tasks common to all airframes, or family of airframes, and
  we call additional methods for specific frames, so that we do not have a code
  that is too heavy.
*/
void ConfigVehicleTypeWidget::updateObjectsFromWidgets()
{
    ConfigTaskWidget::updateObjectsFromWidgets();

    MixerSettings *mixerSettings = MixerSettings::GetInstance(getObjectManager());
    Q_ASSERT(mixerSettings);

    QPointer<VehicleConfig> vconfig = new VehicleConfig();

    frameType = SystemSettings::AIRFRAMETYPE_CUSTOM; //Set airframe type default to "Custom"

    if (m_aircraft->aircraftType->currentText() == "Fixed Wing") {
        frameType = m_fixedwing->updateConfigObjectsFromWidgets();
    }
    else if (m_aircraft->aircraftType->currentText() == "Multirotor") {
         frameType = m_multirotor->updateConfigObjectsFromWidgets();
    }
    else if (m_aircraft->aircraftType->currentText() == "Helicopter") {
         frameType = m_heli->updateConfigObjectsFromWidgets();
    }
    else if (m_aircraft->aircraftType->currentText() == "Ground") {
         frameType = m_groundvehicle->updateConfigObjectsFromWidgets();
    }
    else {
        vconfig->setThrottleCurve(mixerSettings, MixerSettings::MIXER1VECTOR_THROTTLECURVE1, m_aircraft->customThrottle1Curve->getCurve());
        vconfig->setThrottleCurve(mixerSettings, MixerSettings::MIXER1VECTOR_THROTTLECURVE2, m_aircraft->customThrottle2Curve->getCurve());

        // Update the table:
        for (int channel=0; channel<(int)(VehicleConfig::CHANNEL_NUMELEM); channel++) {
            QComboBox* q = (QComboBox*)m_aircraft->customMixerTable->cellWidget(0,channel);
            if(q->currentText()=="Disabled")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_DISABLED);
            else if(q->currentText()=="Motor")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_MOTOR);
            else if(q->currentText()=="Servo")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_SERVO);
            else if(q->currentText()=="CameraRoll")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_CAMERAROLL);
            else if(q->currentText()=="CameraPitch")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_CAMERAPITCH);
            else if(q->currentText()=="CameraYaw")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_CAMERAYAW);
            else if(q->currentText()=="Accessory0")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_ACCESSORY0);
            else if(q->currentText()=="Accessory1")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_ACCESSORY1);
            else if(q->currentText()=="Accessory2")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_ACCESSORY2);
            else if(q->currentText()=="Accessory3")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_ACCESSORY3);
            else if(q->currentText()=="Accessory4")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_ACCESSORY4);
            else if(q->currentText()=="Accessory5")
                vconfig->setMixerType(mixerSettings,channel,MixerSettings::MIXER1TYPE_ACCESSORY5);

            vconfig->setMixerVectorValue(mixerSettings,channel,MixerSettings::MIXER1VECTOR_THROTTLECURVE1,
                                            m_aircraft->customMixerTable->item(1,channel)->text().toDouble());
            vconfig->setMixerVectorValue(mixerSettings,channel,MixerSettings::MIXER1VECTOR_THROTTLECURVE2,
                                            m_aircraft->customMixerTable->item(2,channel)->text().toDouble());
            vconfig->setMixerVectorValue(mixerSettings,channel,MixerSettings::MIXER1VECTOR_ROLL,
                                            m_aircraft->customMixerTable->item(3,channel)->text().toDouble());
            vconfig->setMixerVectorValue(mixerSettings,channel,MixerSettings::MIXER1VECTOR_PITCH,
                                            m_aircraft->customMixerTable->item(4,channel)->text().toDouble());
            vconfig->setMixerVectorValue(mixerSettings,channel,MixerSettings::MIXER1VECTOR_YAW,
                                            m_aircraft->customMixerTable->item(5,channel)->text().toDouble());
        }
    }

    // set the airframe type
    SystemSettings *systemSettings = SystemSettings::GetInstance(getObjectManager());
    Q_ASSERT(systemSettings);
    SystemSettings::DataFields systemSettingsData = systemSettings->getData();

    systemSettingsData.AirframeType = frameType;

    systemSettings->setData(systemSettingsData);

    updateCustomAirframeUI();
}

/**
 Opens the wiki from the user's default browser
 */
void ConfigVehicleTypeWidget::openHelp()
{

    QDesktopServices::openUrl( QUrl("https://github.com/TauLabs/TauLabs/wiki/OnlineHelp:-Vehicle-configuration", QUrl::StrictMode) );
}

/**
  Helper function:
  Sets the current index on supplied combobox to index
  if it is within bounds 0 <= index < combobox.count()
 */
void ConfigVehicleTypeWidget::setComboCurrentIndex(QComboBox* box, int index)
{
    if (index >= 0 && index < box->count())
        box->setCurrentIndex(index);
}

void ConfigVehicleTypeWidget::reverseMultirotorMotor(){
    m_multirotor->drawAirframe(frameType);
}


/**
 * @brief ConfigVehicleTypeWidget::on_bnLevelTrim_clicked Attempts to set autopilot level bias
 * values, and processes the success message
 */
void ConfigVehicleTypeWidget::bnLevelTrim_clicked()
{
    QMessageBox msgBox(QMessageBox::Question, tr("Trim level"),
                       tr("Use the transmitter trim to set the autopilot for straight and level flight? (Please see the tooltip for more information.)"),
                       QMessageBox::Yes | QMessageBox::No, this);
    int userChoice = msgBox.exec();

    // If the user cancels, stop here.
    if (userChoice != QMessageBox::Yes)
        return;

    // Call bias set function
    VehicleTrim vehicleTrim;
    VehicleTrim::autopilotLevelBiasMessages ret;
    ret = vehicleTrim.setAutopilotBias();

    // Process return state
    switch (ret){
    case VehicleTrim::AUTOPILOT_LEVEL_FAILED_DUE_TO_MISSING_RECEIVER:
    {
        QMessageBox msgBox(QMessageBox::Critical, tr("No receiver detected"),
                           tr("Transmitter and receiver must be powered on."), QMessageBox::Ok, this);
        msgBox.exec();
        break;
    }
    case VehicleTrim::AUTOPILOT_LEVEL_FAILED_DUE_TO_ARMED_STATE:
    {
        QMessageBox msgBox(QMessageBox::Critical, tr("Vehicle armed"),
                           tr("The autopilot must be disarmed first."), QMessageBox::Ok, this);
        msgBox.exec();
        break;
    }
    case VehicleTrim::AUTOPILOT_LEVEL_FAILED_DUE_TO_FLIGHTMODE:
    {
        QMessageBox msgBox(QMessageBox::Critical, tr("Vehicle not in Stabilized mode"),
                           tr("The autopilot must be in Leveling, Stabilized1, Stabilized2, or Stabilized3 mode."), QMessageBox::Ok, this);
        msgBox.exec();
        break;
    }
    case VehicleTrim::AUTOPILOT_LEVEL_FAILED_DUE_TO_STABILIZATIONMODE:
    {
        QMessageBox msgBox(QMessageBox::Critical, tr("Incorrect roll and pitch stabilization modes."),
                           tr("Both roll and pitch must be in Attitude stabilization mode."), QMessageBox::Ok, this);
        msgBox.exec();
        break;
    }
    case VehicleTrim::AUTOPILOT_LEVEL_SUCCESS:
        QMessageBox msgBox(QMessageBox::Information, tr("Trim updated"),
                           tr("Trim successfully updated, please reset the transmitter's trim to zero and be sure to configure stabilization settings to use Attitude mode."), QMessageBox::Ok, this);
        msgBox.exec();

        // Set tab as dirty (i.e. having unsaved changes).
        setDirty(true);

        break;
    }
}


/**
 * @brief ConfigVehicleTypeWidget::on_bnServoTrim_clicked Attempts to set actuator trim
 * values, and processes the success message
 */
void ConfigVehicleTypeWidget::bnServoTrim_clicked()
{
    QMessageBox msgBox(QMessageBox::Question, tr("Trim servos"),
                       "Use the transmitter trim to set servos for wings-level, constant-speed flight? (Please see the tooltip for more information.)",
                       QMessageBox::Ok | QMessageBox::Cancel, this);
    int cancelAction = msgBox.exec();

    // If the user cancels, stop here.
    if (cancelAction != QMessageBox::Ok)
        return;

    // Call servo trim function
    VehicleTrim vehicleTrim;
    VehicleTrim::actuatorTrimMessages ret;
    ret = vehicleTrim.setTrimActuators();

    // Process return state
    switch (ret){
    case VehicleTrim::ACTUATOR_TRIM_FAILED_DUE_TO_MISSING_RECEIVER:
    {
        QMessageBox msgBox(QMessageBox::Critical, tr("No receiver detected"),
                           "Transmitter and receiver must be powered on.", QMessageBox::Ok, this);
        msgBox.exec();
        break;
    }
    case VehicleTrim::ACTUATOR_TRIM_FAILED_DUE_TO_FLIGHTMODE:
    {
        QMessageBox msgBox(QMessageBox::Critical, tr("Vehicle not in manual mode"),
                           "The autopilot must be in manual flight mode.", QMessageBox::Ok, this);
        msgBox.exec();
        break;
    }
    case VehicleTrim::ACTUATOR_TRIM_SUCCESS:
        QMessageBox msgBox(QMessageBox::Information, tr("Trim updated"),
                           "Servo trim successfully updated, please reset the transmitter's trim before flying.", QMessageBox::Ok, this);
        msgBox.exec();

        // Set tab as dirty (i.e. having unsaved changes).
        setDirty(true);

        break;
    }

}


/**
 * @brief ConfigVehicleTypeWidget::addToDirtyMonitor Adds the UI widgets to a list of widgets
 * monitored for any changes
 */
void ConfigVehicleTypeWidget::addToDirtyMonitor()
{
    addWidget(m_aircraft->customMixerTable);
    addWidget(m_aircraft->customThrottle1Curve->getCurveWidget());
    addWidget(m_aircraft->customThrottle2Curve->getCurveWidget());
    addWidget(m_aircraft->multiThrottleCurve->getCurveWidget());
    addWidget(m_aircraft->fixedWingThrottle->getCurveWidget());
    addWidget(m_aircraft->fixedWingType);
    addWidget(m_aircraft->groundVehicleThrottle1->getCurveWidget());
    addWidget(m_aircraft->groundVehicleThrottle2->getCurveWidget());
    addWidget(m_aircraft->groundVehicleType);
    addWidget(m_aircraft->multirotorFrameType);
    addWidget(m_aircraft->multiMotorChannelBox1);
    addWidget(m_aircraft->multiMotorChannelBox2);
    addWidget(m_aircraft->multiMotorChannelBox3);
    addWidget(m_aircraft->multiMotorChannelBox4);
    addWidget(m_aircraft->multiMotorChannelBox5);
    addWidget(m_aircraft->multiMotorChannelBox6);
    addWidget(m_aircraft->multiMotorChannelBox7);
    addWidget(m_aircraft->multiMotorChannelBox8);
    addWidget(m_aircraft->mrPitchMixLevel);
    addWidget(m_aircraft->mrRollMixLevel);
    addWidget(m_aircraft->mrYawMixLevel);
    addWidget(m_aircraft->triYawChannelBox);
    addWidget(m_aircraft->aircraftType);
    addWidget(m_aircraft->fwEngineChannelBox);
    addWidget(m_aircraft->fwAileron1ChannelBox);
    addWidget(m_aircraft->fwAileron2ChannelBox);
    addWidget(m_aircraft->fwElevator1ChannelBox);
    addWidget(m_aircraft->fwElevator2ChannelBox);
    addWidget(m_aircraft->fwRudder1ChannelBox);
    addWidget(m_aircraft->fwRudder2ChannelBox);
    addWidget(m_aircraft->elevonSlider1);
    addWidget(m_aircraft->elevonSlider2);
    addWidget(m_heli->m_ccpm->ccpmType);
    addWidget(m_heli->m_ccpm->ccpmTailChannel);
    addWidget(m_heli->m_ccpm->ccpmEngineChannel);
    addWidget(m_heli->m_ccpm->ccpmServoWChannel);
    addWidget(m_heli->m_ccpm->ccpmServoXChannel);
    addWidget(m_heli->m_ccpm->ccpmServoYChannel);
    addWidget(m_heli->m_ccpm->ccpmSingleServo);
    addWidget(m_heli->m_ccpm->ccpmServoZChannel);
    addWidget(m_heli->m_ccpm->ccpmAngleW);
    addWidget(m_heli->m_ccpm->ccpmAngleX);
    addWidget(m_heli->m_ccpm->ccpmCorrectionAngle);
    addWidget(m_heli->m_ccpm->ccpmAngleZ);
    addWidget(m_heli->m_ccpm->ccpmAngleY);
    addWidget(m_heli->m_ccpm->ccpmCollectivePassthrough);
    addWidget(m_heli->m_ccpm->ccpmLinkRoll);
    addWidget(m_heli->m_ccpm->ccpmLinkCyclic);
    addWidget(m_heli->m_ccpm->ccpmRevoSlider);
    addWidget(m_heli->m_ccpm->ccpmREVOspinBox);
    addWidget(m_heli->m_ccpm->ccpmCollectiveSlider);
    addWidget(m_heli->m_ccpm->ccpmCollectivespinBox);
    addWidget(m_heli->m_ccpm->ccpmCollectiveScale);
    addWidget(m_heli->m_ccpm->ccpmCollectiveScaleBox);
    addWidget(m_heli->m_ccpm->ccpmCyclicScale);
    addWidget(m_heli->m_ccpm->ccpmPitchScale);
    addWidget(m_heli->m_ccpm->ccpmPitchScaleBox);
    addWidget(m_heli->m_ccpm->ccpmRollScale);
    addWidget(m_heli->m_ccpm->ccpmRollScaleBox);
    addWidget(m_heli->m_ccpm->SwashLvlPositionSlider);
    addWidget(m_heli->m_ccpm->SwashLvlPositionSpinBox);
    addWidget(m_heli->m_ccpm->ThrottleCurve->getCurveWidget());
    addWidget(m_heli->m_ccpm->PitchCurve->getCurveWidget());
    addWidget(m_heli->m_ccpm->ccpmAdvancedSettingsTable);
}

