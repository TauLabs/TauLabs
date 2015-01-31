/**
 ******************************************************************************
 *
 * @file       configstabilizationwidget.h
 * @author     E. Lafargue & The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief The Configuration Gadget used to update settings in the firmware
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
#include "configstabilizationwidget.h"
#include "convertmwrate.h"

#include <QDebug>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <QDesktopServices>
#include <QUrl>
#include <QList>


#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>


ConfigStabilizationWidget::ConfigStabilizationWidget(QWidget *parent) : ConfigTaskWidget(parent)
{
    m_stabilization = new Ui_StabilizationWidget();
    m_stabilization->setupUi(this);


    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if(!settings->useExpertMode() || true)
        m_stabilization->saveStabilizationToRAM_6->setVisible(false);
    m_stabilization->saveStabilizationToRAM_6->setVisible(true);

    

    autoLoadWidgets();
    realtimeUpdates=new QTimer(this);
    connect(m_stabilization->realTimeUpdates_6,SIGNAL(stateChanged(int)),this,SLOT(realtimeUpdatesSlot(int)));
    connect(realtimeUpdates,SIGNAL(timeout()),this,SLOT(apply()));

    connect(m_stabilization->checkBox_7,SIGNAL(stateChanged(int)),this,SLOT(linkCheckBoxes(int)));
    connect(m_stabilization->checkBox_2,SIGNAL(stateChanged(int)),this,SLOT(linkCheckBoxes(int)));
    connect(m_stabilization->checkBox_8,SIGNAL(stateChanged(int)),this,SLOT(linkCheckBoxes(int)));
    connect(m_stabilization->checkBox_3,SIGNAL(stateChanged(int)),this,SLOT(linkCheckBoxes(int)));

    connect(this,SIGNAL(widgetContentsChanged(QWidget*)),this,SLOT(processLinkedWidgets(QWidget*)));

    connect(m_stabilization->calculateMW, SIGNAL(clicked()), this, SLOT(showMWRateConvertDialog()));

    disableMouseWheelEvents();

    connect(this,SIGNAL(autoPilotConnected()),this,SLOT(applyRateLimits()));
}


ConfigStabilizationWidget::~ConfigStabilizationWidget()
{
    // Do nothing
}

void ConfigStabilizationWidget::realtimeUpdatesSlot(int value)
{
    m_stabilization->realTimeUpdates_6->setCheckState((Qt::CheckState)value);
    if(value==Qt::Checked && !realtimeUpdates->isActive())
        realtimeUpdates->start(300);
    else if(value==Qt::Unchecked)
        realtimeUpdates->stop();
}

void ConfigStabilizationWidget::linkCheckBoxes(int value)
{
    if(sender()== m_stabilization->checkBox_7)
        m_stabilization->checkBox_3->setCheckState((Qt::CheckState)value);
    else if(sender()== m_stabilization->checkBox_3)
        m_stabilization->checkBox_7->setCheckState((Qt::CheckState)value);
    else if(sender()== m_stabilization->checkBox_8)
        m_stabilization->checkBox_2->setCheckState((Qt::CheckState)value);
    else if(sender()== m_stabilization->checkBox_2)
        m_stabilization->checkBox_8->setCheckState((Qt::CheckState)value);
}

void ConfigStabilizationWidget::processLinkedWidgets(QWidget * widget)
{
    if(m_stabilization->checkBox_7->checkState()==Qt::Checked)
    {
        if(widget== m_stabilization->RateRollKp_2)
        {
            m_stabilization->RatePitchKp->setValue(m_stabilization->RateRollKp_2->value());
        }
        else if(widget== m_stabilization->RateRollKi_2)
        {
            m_stabilization->RatePitchKi->setValue(m_stabilization->RateRollKi_2->value());
        }
        else if(widget== m_stabilization->RateRollILimit_2)
        {
            m_stabilization->RatePitchILimit->setValue(m_stabilization->RateRollILimit_2->value());
        }
        else if(widget== m_stabilization->RatePitchKp)
        {
            m_stabilization->RateRollKp_2->setValue(m_stabilization->RatePitchKp->value());
        }
        else if(widget== m_stabilization->RatePitchKi)
        {
            m_stabilization->RateRollKi_2->setValue(m_stabilization->RatePitchKi->value());
        }
        else if(widget== m_stabilization->RatePitchILimit)
        {
            m_stabilization->RateRollILimit_2->setValue(m_stabilization->RatePitchILimit->value());
        }
        else if(widget== m_stabilization->RollRateKd)
        {
            m_stabilization->PitchRateKd->setValue(m_stabilization->RollRateKd->value());
        }
        else if(widget== m_stabilization->PitchRateKd)
        {
            m_stabilization->RollRateKd->setValue(m_stabilization->PitchRateKd->value());
        }
    }
    if(m_stabilization->checkBox_8->checkState()==Qt::Checked)
    {
        if(widget== m_stabilization->AttitudeRollKp)
        {
            m_stabilization->AttitudePitchKp_2->setValue(m_stabilization->AttitudeRollKp->value());
        }
        else if(widget== m_stabilization->AttitudeRollKi)
        {
            m_stabilization->AttitudePitchKi_2->setValue(m_stabilization->AttitudeRollKi->value());
        }
        else if(widget== m_stabilization->AttitudeRollILimit)
        {
            m_stabilization->AttitudePitchILimit_2->setValue(m_stabilization->AttitudeRollILimit->value());
        }
        else if(widget== m_stabilization->AttitudePitchKp_2)
        {
            m_stabilization->AttitudeRollKp->setValue(m_stabilization->AttitudePitchKp_2->value());
        }
        else if(widget== m_stabilization->AttitudePitchKi_2)
        {
            m_stabilization->AttitudeRollKi->setValue(m_stabilization->AttitudePitchKi_2->value());
        }
        else if(widget== m_stabilization->AttitudePitchILimit_2)
        {
            m_stabilization->AttitudeRollILimit->setValue(m_stabilization->AttitudePitchILimit_2->value());
        }
    }

    // sync the multiwii rate settings
    if (m_stabilization->cb_linkMwRollPitch->checkState()==Qt::Checked) {
        if (widget == m_stabilization->MWRatePitchKp)
            m_stabilization->MWRateRollKp->setValue(m_stabilization->MWRatePitchKp->value());
        else if (widget == m_stabilization->MWRateRollKp)
            m_stabilization->MWRatePitchKp->setValue(m_stabilization->MWRateRollKp->value());
        else if (widget == m_stabilization->MWRatePitchKi)
            m_stabilization->MWRateRollKi->setValue(m_stabilization->MWRatePitchKi->value());
        else if (widget == m_stabilization->MWRateRollKi)
            m_stabilization->MWRatePitchKi->setValue(m_stabilization->MWRateRollKi->value());
        else if (widget == m_stabilization->MWRatePitchKd)
            m_stabilization->MWRateRollKd->setValue(m_stabilization->MWRatePitchKd->value());
        else if (widget == m_stabilization->MWRateRollKd)
            m_stabilization->MWRatePitchKd->setValue(m_stabilization->MWRateRollKd->value());
    }
}

void ConfigStabilizationWidget::applyRateLimits()
{
    Core::IBoardType *board = getObjectUtilManager()->getBoardType();

    double maxRate = 500; // Default to slowest across boards
    if (board)
        maxRate = board->queryMaxGyroRate() * 0.85;

    m_stabilization->fullStickRateRoll->setMaximum(maxRate);
    m_stabilization->fullStickRatePitch->setMaximum(maxRate);
    m_stabilization->fullStickRateYaw->setMaximum(maxRate);
    m_stabilization->maxRateAttRoll->setMaximum(maxRate);
    m_stabilization->maxRateAttPitch->setMaximum(maxRate);
    m_stabilization->maxRateAttYaw->setMaximum(maxRate);
}

void ConfigStabilizationWidget::showMWRateConvertDialog()
{
    ConvertMWRate *dialog = new ConvertMWRate(this);

    connect(dialog, SIGNAL(accepted()), this, SLOT(applyMWRateConvertDialog()));
    dialog->exec();
}

void ConfigStabilizationWidget::applyMWRateConvertDialog()
{
    ConvertMWRate *dialog = dynamic_cast<ConvertMWRate *>(sender());
    if (dialog) {
        m_stabilization->MWRateRollKp->setValue(dialog->getRollKp());
        m_stabilization->MWRateRollKi->setValue(dialog->getRollKi());
        m_stabilization->MWRateRollKd->setValue(dialog->getRollKd());
        m_stabilization->MWRatePitchKp->setValue(dialog->getPitchKp());
        m_stabilization->MWRatePitchKi->setValue(dialog->getPitchKi());
        m_stabilization->MWRatePitchKd->setValue(dialog->getPitchKd());
        m_stabilization->MWRateYawKp->setValue(dialog->getYawKp());
        m_stabilization->MWRateYawKi->setValue(dialog->getYawKi());
        m_stabilization->MWRateYawKd->setValue(dialog->getYawKd());
    }
}

