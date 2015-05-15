/**
 ******************************************************************************
 *
 * @file       setupwizard.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup SetupWizard Setup Wizard
 * @{
 * @brief A Wizard to make the initial setup easy for everyone.
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

#include "setupwizard.h"
#include "pages/tlstartpage.h"
#include "pages/tlendpage.h"
#include "pages/boardtype_unknown.h"
#include "pages/controllerpage.h"
#include "pages/vehiclepage.h"
#include "pages/multipage.h"
#include "pages/fixedwingpage.h"
#include "pages/helipage.h"
#include "pages/surfacepage.h"
#include "pages/inputpage.h"
#include "pages/inputpage_notsupported.h"
#include "pages/outputpage.h"
#include "pages/biascalibrationpage.h"
#include "pages/summarypage.h"
#include "pages/savepage.h"
#include "pages/notyetimplementedpage.h"
#include "pages/rebootpage.h"
#include "pages/outputcalibrationpage.h"
#include "extensionsystem/pluginmanager.h"
#include "vehicleconfigurationhelper.h"
#include "actuatorsettings.h"
#include "pages/autoupdatepage.h"
#include "uploader/uploadergadgetfactory.h"

using namespace uploader;

SetupWizard::SetupWizard(QWidget *parent) : QWizard(parent), VehicleConfigurationSource(),
    m_controllerType(NULL),
    m_vehicleType(VEHICLE_UNKNOWN), m_inputType(Core::IBoardType::INPUT_TYPE_UNKNOWN), m_escType(ESC_UNKNOWN),
    m_calibrationPerformed(false), m_restartNeeded(false), m_connectionManager(0)
{
    setWindowTitle(tr("Tau Labs Setup Wizard"));
    setOption(QWizard::IndependentPages, false);
    for (quint16 i = 0; i < ActuatorSettings::CHANNELMAX_NUMELEM; i++) {
        m_actuatorSettings << actuatorChannelSettings();
    }
    setWizardStyle(QWizard::ModernStyle);
    setMinimumSize(600, 500);
    resize(600, 500);
    createPages();
}

int SetupWizard::nextId() const
{
    switch (currentId()) {
    case PAGE_START:
        if (canAutoUpdate()) {
            return PAGE_UPDATE;
        } else {
            return PAGE_CONTROLLER;
        }
    case PAGE_UPDATE:
        return PAGE_CONTROLLER;

    case PAGE_CONTROLLER:
    {
        Core::IBoardType* type = getControllerType();
        if (type != NULL && type->isInputConfigurationSupported())
            return PAGE_INPUT;
        else if (type != NULL)
            return PAGE_INPUT_NOT_SUPPORTED;
        else
            return PAGE_BOARDTYPE_UNKNOWN;
    }
    case PAGE_VEHICLES:
    {
        switch (getVehicleType()) {
        case VEHICLE_MULTI:
            return PAGE_MULTI;

        case VEHICLE_FIXEDWING:
            return PAGE_FIXEDWING;

        case VEHICLE_HELI:
            return PAGE_HELI;

        case VEHICLE_SURFACE:
            return PAGE_SURFACE;

        default:
            return PAGE_NOTYETIMPLEMENTED;
        }
    }
    case PAGE_MULTI:
        return PAGE_OUTPUT;

    case PAGE_INPUT:
        if (isRestartNeeded()) {
            saveHardwareSettings();
            return PAGE_REBOOT;
        } else {
            return PAGE_VEHICLES;
        }
    case PAGE_REBOOT:
        return PAGE_VEHICLES;

    case PAGE_OUTPUT:
        return PAGE_SUMMARY;

    case PAGE_BIAS_CALIBRATION:
        return PAGE_OUTPUT_CALIBRATION;

    case PAGE_OUTPUT_CALIBRATION:
        return PAGE_SAVE;

    case PAGE_SUMMARY:
    {
        Core::IBoardType* type = getControllerType();
        if (type != NULL)
            return PAGE_BIAS_CALIBRATION;
        else
            return PAGE_NOTYETIMPLEMENTED;
    }
    case PAGE_SAVE:
        return PAGE_END;

    case PAGE_INPUT_NOT_SUPPORTED:
        // The user is informed on this page to configure manually and restart the wizard
        // but if they advance assume they have configured the input.
        return PAGE_VEHICLES;

    case PAGE_BOARDTYPE_UNKNOWN:
    case PAGE_NOTYETIMPLEMENTED:
        return PAGE_END;

    default:
        return -1;
    }
}

QString SetupWizard::getSummaryText()
{
    QString summary = "";

    summary.append("<b>").append(tr("Controller type: ")).append("</b>");
    Core::IBoardType* type = getControllerType();
    if (type != NULL)
        summary.append(type->shortName());
    else
        summary.append(tr("Unknown"));

    summary.append("<br>");
    summary.append("<b>").append(tr("Vehicle type: ")).append("</b>");
    switch (getVehicleType()) {
    case VEHICLE_MULTI:
        summary.append(tr("Multirotor"));

        summary.append("<br>");
        summary.append("<b>").append(tr("Vehicle sub type: ")).append("</b>");
        switch (getVehicleSubType()) {
        case SetupWizard::MULTI_ROTOR_TRI_Y:
            summary.append(tr("Tricopter"));
            break;
        case SetupWizard::MULTI_ROTOR_QUAD_X:
            summary.append(tr("Quadcopter X"));
            break;
        case SetupWizard::MULTI_ROTOR_QUAD_PLUS:
            summary.append(tr("Quadcopter +"));
            break;
        case SetupWizard::MULTI_ROTOR_HEXA:
            summary.append(tr("Hexacopter"));
            break;
        case SetupWizard::MULTI_ROTOR_HEXA_COAX_Y:
            summary.append(tr("Hexacopter Coax (Y6)"));
            break;
        case SetupWizard::MULTI_ROTOR_HEXA_H:
            summary.append(tr("Hexacopter X"));
            break;
        case SetupWizard::MULTI_ROTOR_OCTO:
            summary.append(tr("Octocopter"));
            break;
        case SetupWizard::MULTI_ROTOR_OCTO_COAX_X:
            summary.append(tr("Octocopter Coax X"));
            break;
        case SetupWizard::MULTI_ROTOR_OCTO_COAX_PLUS:
            summary.append(tr("Octocopter Coax +"));
            break;
        case SetupWizard::MULTI_ROTOR_OCTO_V:
            summary.append(tr("Octocopter V"));
            break;
        default:
            summary.append(tr("Unknown"));
            break;
        }

        break;
    case VEHICLE_FIXEDWING:
        summary.append(tr("Fixed wing"));
        break;
    case VEHICLE_HELI:
        summary.append(tr("Helicopter"));
        break;
    case VEHICLE_SURFACE:
        summary.append(tr("Surface vehicle"));
        break;
    default:
        summary.append(tr("Unknown"));
    }

    summary.append("<br>");
    summary.append("<b>").append(tr("Input type: ")).append("</b>");
    switch (getInputType()) {
    case Core::IBoardType::INPUT_TYPE_PWM:
        summary.append(tr("PWM (One cable per channel)"));
        break;
    case Core::IBoardType::INPUT_TYPE_PPM:
        summary.append(tr("PPM (One cable for all channels)"));
        break;
    case Core::IBoardType::INPUT_TYPE_SBUS:
        summary.append(tr("Futaba S.Bus"));
        break;
    case Core::IBoardType::INPUT_TYPE_DSM:
        summary.append(tr("Spektrum satellite (DSM)"));
        break;
    case Core::IBoardType::INPUT_TYPE_HOTTSUMD:
        summary.append(tr("Graupner HoTT (SUMD)"));
        break;
    case Core::IBoardType::INPUT_TYPE_HOTTSUMH:
        summary.append(tr("Graupner HoTT (SUMH)"));
        break;
    default:
        summary.append(tr("Unknown"));
    }

    summary.append("<br>");
    summary.append("<b>").append(tr("ESC type: ")).append("</b>");
    switch (getESCType()) {
    case ESC_LEGACY:
        summary.append(tr("Legacy ESC (50 Hz)"));
        break;
    case ESC_RAPID:
        summary.append(tr("Rapid ESC (400 Hz)"));
        break;
    case ESC_ONESHOT:
        summary.append(tr("OneShot (PWMSync + 125-250us)"));
        break;
    default:
        summary.append(tr("Unknown"));
    }

    /*
       summary.append("<br>");
       summary.append("<b>").append(tr("Reboot required: ")).append("</b>");
       summary.append(isRestartNeeded() ? tr("<font color='red'>Yes</font>") : tr("<font color='green'>No</font>"));
     */
    return summary;
}

void SetupWizard::createPages()
{
    setPage(PAGE_START, new TLStartPage(this));
    setPage(PAGE_UPDATE, new AutoUpdatePage(this));
    setPage(PAGE_CONTROLLER, new ControllerPage(this));
    setPage(PAGE_VEHICLES, new VehiclePage(this));
    setPage(PAGE_MULTI, new MultiPage(this));
    setPage(PAGE_FIXEDWING, new FixedWingPage(this));
    setPage(PAGE_HELI, new HeliPage(this));
    setPage(PAGE_SURFACE, new SurfacePage(this));
    setPage(PAGE_INPUT, new InputPage(this));
    setPage(PAGE_INPUT_NOT_SUPPORTED, new InputPageNotSupported(this));
    setPage(PAGE_OUTPUT, new OutputPage(this));
    setPage(PAGE_BIAS_CALIBRATION, new BiasCalibrationPage(this));
    setPage(PAGE_OUTPUT_CALIBRATION, new OutputCalibrationPage(this));
    setPage(PAGE_SUMMARY, new SummaryPage(this));
    setPage(PAGE_SAVE, new SavePage(this));
    setPage(PAGE_REBOOT, new RebootPage(this));
    setPage(PAGE_NOTYETIMPLEMENTED, new NotYetImplementedPage(this));
    setPage(PAGE_BOARDTYPE_UNKNOWN, new BoardtypeUnknown(this));
    setPage(PAGE_END, new TLEndPage(this));

    setStartId(PAGE_START);

    connect(button(QWizard::CustomButton1), SIGNAL(clicked()), this, SLOT(customBackClicked()));
    setButtonText(QWizard::CustomButton1, buttonText(QWizard::BackButton));
    QList<QWizard::WizardButton> button_layout;
    button_layout << QWizard::Stretch << QWizard::CustomButton1 << QWizard::NextButton << QWizard::CancelButton << QWizard::FinishButton;
    setButtonLayout(button_layout);
    connect(this, SIGNAL(currentIdChanged(int)), this, SLOT(pageChanged(int)));
}

void SetupWizard::customBackClicked()
{
    if (currentId() == PAGE_OUTPUT_CALIBRATION) {
        static_cast<OutputCalibrationPage *>(currentPage())->customBackClicked();
    } else {
        back();
    }
}

void SetupWizard::pageChanged(int currId)
{
    button(QWizard::CustomButton1)->setVisible(currId != PAGE_START);
    button(QWizard::CancelButton)->setVisible(currId != PAGE_END);
}

bool SetupWizard::saveHardwareSettings() const
{
    VehicleConfigurationHelper helper(const_cast<SetupWizard *>(this));

    return helper.setupHardwareSettings();
}

/**
 * @brief SetupWizard::canAutoUpdate determine if build can autoupdated
 *
 * This checks for the firmware resource file being existing to see if
 * auto-updating is even an option.
 *
 * @return true if auto-update can be attempted
 */
bool SetupWizard::canAutoUpdate() const
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();

    Q_ASSERT(pm);
    UploaderGadgetFactory *uploader    = pm->getObject<UploaderGadgetFactory>();
    Q_ASSERT(uploader);
    return uploader->isAutoUpdateCapable();
}
