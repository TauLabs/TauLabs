/**
 ******************************************************************************
 * @file       hwfieldselector.cpp
 * @author     Tau Labs, http://github.com/TauLabs, Copyright (C) 2012-2013.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief A selector widget for hardware configuration field options
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

#include "hwfieldselector.h"

HwFieldSelector::HwFieldSelector(QWidget *parent) :
    QWidget(parent)
{
    ui = new Ui::HwFieldSelector();
    ui->setupUi(this);
}

void HwFieldSelector::setUavoField(UAVObjectField *field)
{
    ui->Name->setText(field->getName());
    ui->Selection->addItems(field->getOptions());
}

//! Accessor for the QComboBox
QComboBox *HwFieldSelector::getCombo()
{
    return ui->Selection;
}
