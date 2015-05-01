/**
 ******************************************************************************
 *
 * @file       outputchannelform.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Servo output configuration form for the config output gadget
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

#include <limits>

#include "outputchannelform.h"
#include "configoutputwidget.h"
#include "actuatorsettings.h"

OutputChannelForm::OutputChannelForm(const int index, QWidget *parent, const bool showLegend) :
        ConfigTaskWidget(parent),
        ui(),
        m_index(index),
        m_inChannelTest(false)
{
    ui.setupUi(this);
    if(!showLegend)
    {
        // Remove legend
        QGridLayout *grid_layout = dynamic_cast<QGridLayout*>(layout());
        Q_ASSERT(grid_layout);
        for (int col = 0; col < grid_layout->columnCount(); col++)
        { // remove every item in first row
            QLayoutItem *item = grid_layout->itemAtPosition(0, col);
            if (!item) continue;
            // get widget from layout item
            QWidget *legend_widget = item->widget();
            if (!legend_widget) continue;
            // delete widget
            grid_layout->removeWidget(legend_widget);
            delete legend_widget;
        }
    }

    // The convention is Channel 1 to Channel 10.
    ui.actuatorNumber->setText(QString("%1:").arg(m_index+1));

    // Register for ActuatorSettings changes:
    connect(ui.actuatorMin, SIGNAL(editingFinished()), this, SLOT(setChannelRange()));
    connect(ui.actuatorMax, SIGNAL(editingFinished()), this, SLOT(setChannelRange()));
    connect(ui.pb_reverseActuator, SIGNAL(clicked()), this, SLOT(reverseChannel()));

    // Connect the channel out sliders to our signal in order to send updates in test mode
    connect(ui.actuatorNeutral, SIGNAL(valueChanged(int)), this, SLOT(sendChannelTest(int)));

    // Connect UI elements to dirty/clean (i.e. changed/unchanged) signal/slot
    connect(ui.actuatorMin, SIGNAL(valueChanged(int)), this, SLOT(notifyFormChanged()));
    connect(ui.actuatorMax, SIGNAL(valueChanged(int)), this, SLOT(notifyFormChanged()));
    connect(ui.actuatorNeutral, SIGNAL(sliderReleased()), this, SLOT(notifyFormChanged()));
    connect(ui.actuatorType, SIGNAL(currentIndexChanged(int)), this, SLOT(notifyFormChanged()));

    ui.actuatorLink->setChecked(false);
    connect(ui.actuatorLink, SIGNAL(toggled(bool)), this, SLOT(linkToggled(bool)));

    // Trigger when autopilot is connected
    connect(this, SIGNAL(autoPilotConnected()), this, SLOT(onAutopilotConnect()));

    // Get UAVObject and connect
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    connect(ActuatorSettings::GetInstance(objManager), SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updateMaxSpinboxValue(UAVObject*)));

    disableMouseWheelEvents();
}

OutputChannelForm::~OutputChannelForm()
{
    // Do nothing
}

/**
 * Restrict UI to protect users from accidental misuse.
 */
void OutputChannelForm::enableChannelTest(bool state)
{
    if (m_inChannelTest == state)
        return;
    m_inChannelTest = state;

    if(m_inChannelTest)
    {
        // Prevent users from changing the minimum & maximum ranges while
        // moving the sliders. Thanks Ivan for the tip :)
        ui.actuatorMin->setEnabled(false);
        ui.actuatorMax->setEnabled(false);
        ui.pb_reverseActuator->setEnabled(false);
        ui.actuatorType->setEnabled(false);
    }
    else
    {
        ui.actuatorMin->setEnabled(true);
        ui.actuatorMax->setEnabled(true);
        ui.pb_reverseActuator->setEnabled(true);
        ui.actuatorType->setEnabled(true);
    }
}


/**
 * Toggles the channel linked state for use in testing mode
 */
void OutputChannelForm::linkToggled(bool state)
{
    Q_UNUSED(state)

    if (!m_inChannelTest)
        return;	// we are not in Test Output mode

    // find the minimum slider value for the linked ones
    if (!parent()) return;
    int min = 10000;
    int linked_count = 0;
    QList<OutputChannelForm*> outputChannelForms = parent()->findChildren<OutputChannelForm*>();
    // set the linked channels of the parent widget to the same value
    foreach(OutputChannelForm *outputChannelForm, outputChannelForms)
    {
        if (!outputChannelForm->ui.actuatorLink->checkState())
            continue;
        if (this == outputChannelForm)
            continue;
        int value = outputChannelForm->ui.actuatorNeutral->value();
        if(min > value) min = value;
        linked_count++;
    }

    if (linked_count <= 0)
        return;		// no linked channels

    // set the linked channels to the same value
    foreach(OutputChannelForm *outputChannelForm, outputChannelForms)
    {
        if (!outputChannelForm->ui.actuatorLink->checkState())
            continue;
        outputChannelForm->ui.actuatorNeutral->setValue(min);
    }
}

/**
 * Set maximal channel value.
 */
void OutputChannelForm::setMax(int maximum)
{
    setMinmax(ui.actuatorMin->value(), maximum);
}

/**
 * Set minimal channel value.
 */
void OutputChannelForm::setMin(int minimum)
{
    setMinmax(minimum, ui.actuatorMax->value());
}

/**
 * Set minimal and maximal channel value.
 */
void OutputChannelForm::setMinmax(int minimum, int maximum)
{
    ui.actuatorMin->setValue(minimum);
    ui.actuatorMax->setValue(maximum);
    setChannelRange();
}

/**
 * Set neutral of channel.
 */
void OutputChannelForm::setNeutral(int value)
{
    ui.actuatorNeutral->setValue(value);
}

/**
 * Set type of channel.
 */
void OutputChannelForm::setType(int value)
{
    ui.actuatorType->setCurrentIndex(value);
}

int OutputChannelForm::type() const
{
    return ui.actuatorType->currentIndex();
}

/**
 * Set the channel assignment label.
 */
void OutputChannelForm::setAssignment(const QString &assignment)
{
    ui.actuatorName->setText(assignment);
    QFontMetrics metrics(ui.actuatorName->font());
    int width=metrics.width(assignment)+1;
    foreach(OutputChannelForm * form,parent()->findChildren<OutputChannelForm*>())
    {
        if(form==this)
            continue;
        if(form->ui.actuatorName->minimumSize().width()<width)
            form->ui.actuatorName->setMinimumSize(width,0);
        else
            width=form->ui.actuatorName->minimumSize().width();
    }
    ui.actuatorName->setMinimumSize(width,0);
}

/**
 * Sets the minimum/maximum value of the channel output sliders.
 * Have to do it here because setMinimum is not a slot.
 *
 * One added trick: if the slider is at its min when the value
 * is changed, then keep it on the min.
 */
void OutputChannelForm::setChannelRange()
{
    if (ui.actuatorMin->value() < ui.actuatorMax->value())
    {
        ui.actuatorNeutral->setRange(ui.actuatorMin->value(), ui.actuatorMax->value());
        ui.actuatorNeutral->setEnabled(true);
    }
    else if (ui.actuatorMin->value() > ui.actuatorMax->value())
    {
        ui.actuatorNeutral->setRange(ui.actuatorMax->value(), ui.actuatorMin->value());
        ui.actuatorNeutral->setEnabled(true);
    } else {
        // when the min and max is equal, disable this slider to prevent crash
        // from Qt bug: https://bugreports.qt.io/browse/QTBUG-43398
        ui.actuatorNeutral->setRange(ui.actuatorMin->value()-1, ui.actuatorMin->value()+1);
        ui.actuatorNeutral->setEnabled(false);
	setNeutral(ui.actuatorMin->value());
    }

    // Force a full slider update
    updateSlider();
}

/**
 * Reverses the channel when the checkbox is clicked
 */
void OutputChannelForm::reverseChannel()
{
    // Swap the min & max values
    int temp = ui.actuatorMax->value();
    ui.actuatorMax->setValue(ui.actuatorMin->value());
    ui.actuatorMin->setValue(temp);

    // Force slider update
    setChannelRange();
}


/**
 * Inverts the slider when the output channel is reversed
 */
/**
 * @brief OutputChannelForm::updateSlider
 */
void OutputChannelForm::updateSlider()
{
    // Invert the slider
    if(ui.actuatorMin->value() > ui.actuatorMax->value()) {
        ui.actuatorNeutral->setInvertedAppearance(true);

        // Set the QSlider groove colors so that the fill is on the side of the minimum value
        ui.actuatorNeutral->setProperty("state", "inverted");
    } else {
        ui.actuatorNeutral->setInvertedAppearance(false);

        // Set the QSlider groove colors so that the fill is on the side of the minimum value
        ui.actuatorNeutral->setProperty("state", "normal");
    }

    // Force refresh of style sheet.
    ui.actuatorNeutral->setStyle(QApplication::style());
}


/**
 * Emits the channel value which will be sent to the UAV to move the servo.
 * Returns immediately if we are not in testing mode.
 */
void OutputChannelForm::sendChannelTest(int value)
{
    int in_value = value;

    QSlider *ob = (QSlider *)QObject::sender();
    if (!ob)
        return;

    if (ui.actuatorMin->value() > ui.actuatorMax->value())
            value = ui.actuatorMin->value() - value + ui.actuatorMax->value();	// the channel is reversed

    if (ui.actuatorLink->checkState() && parent())
    {	// the channel is linked to other channels
        QList<OutputChannelForm*> outputChannelForms = parent()->findChildren<OutputChannelForm*>();
        // set the linked channels of the parent widget to the same value
        foreach(OutputChannelForm *outputChannelForm, outputChannelForms)
        {
            if (this == outputChannelForm) continue;
            if (!outputChannelForm->ui.actuatorLink->checkState()) continue;

            int val = in_value;
            if (val < outputChannelForm->ui.actuatorNeutral->minimum())
                val = outputChannelForm->ui.actuatorNeutral->minimum();
            if (val > outputChannelForm->ui.actuatorNeutral->maximum())
                val = outputChannelForm->ui.actuatorNeutral->maximum();

            if (outputChannelForm->ui.actuatorNeutral->value() == val) continue;

            outputChannelForm->ui.actuatorNeutral->setValue(val);
        }
    }

    if (!m_inChannelTest)
        return;	// we are not in Test Output mode

    emit channelChanged(index(), value);
}

/**
 * @brief OutputChannelForm::setUpdated Slot that receives signals indicating the UI is updated
 */
void OutputChannelForm::notifyFormChanged()
{
    // If we are not in Test Output mode, set form as dirty
    if (!m_inChannelTest){
        emit formChanged();
    }
}


void OutputChannelForm::updateMaxSpinboxValue(UAVObject *obj)
{
    Q_UNUSED(obj);

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    ActuatorSettings *actuatorSettings = ActuatorSettings::GetInstance(objManager);
    ActuatorSettings::DataFields actuatorSettingsData = actuatorSettings->getData();

    UAVObjectUtilManager* utilMngr = pm->getObject<UAVObjectUtilManager>();
    Core::IBoardType *board = utilMngr->getBoardType();

    // Check that a board is registered
    if (board == NULL)
        return;

    QVector< QVector<qint32> > channelBanks = board->getChannelBanks();

    for (int i=0; i<channelBanks.size(); i++) {
        QVector<int> channelBank = channelBanks[i];

        // Iterate over each channel...
        foreach(qint32 channel, channelBank) {
            // ... and if there's a match, set the maximum values and return
            if (channel-1 == m_index) {
                if (actuatorSettingsData.ChannelUpdateFreq[i] == 0)
                    return;

                double maxPulseWidth = round(10000000.0 / actuatorSettingsData.ChannelUpdateFreq[i]);

                // Saturate at the UAVO's maximum value
                if (maxPulseWidth > std::numeric_limits<__typeof__(actuatorSettingsData.ChannelMax[0])>::max())
                    maxPulseWidth = std::numeric_limits<__typeof__(actuatorSettingsData.ChannelMax[0])>::max();

                ui.actuatorMin->setMaximum(maxPulseWidth);
                ui.actuatorMax->setMaximum(maxPulseWidth);

                return;
            }
        }
    }
}


/**
 * @brief OutputChannelForm::onAutopilotConnect Triggers a spinbox update. This resolves a race
 * condition by which the ActuatorSettings UAVO could be updated before the board manager
 * loaded the appropriate board settings
 */
void OutputChannelForm::onAutopilotConnect()
{
    // Trigger an update
    updateMaxSpinboxValue((UAVObject *)NULL);
}
