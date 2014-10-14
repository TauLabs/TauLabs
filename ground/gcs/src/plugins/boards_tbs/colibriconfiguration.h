/**
 ******************************************************************************
 *
 * @file       colibriconfiguration.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_TBS TBS boards support Plugin
 * @{
 * @brief Plugin to support boards by TBS
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

#ifndef COLIBRICONFIGURATION_H
#define COLIBRICONFIGURATION_H

#include <QPixmap>
#include "configtaskwidget.h"

namespace Ui {
class ColibriConfiguration;
}

class ColibriConfiguration : public ConfigTaskWidget
{
    Q_OBJECT
    
public:
    explicit ColibriConfiguration(QWidget *parent = 0);
    ~ColibriConfiguration();

private slots:
    void openHelp();
    void refreshValues();
    void widgetsContentsChanged();
    void resizeEvent(QResizeEvent *event);

private:
    Ui::ColibriConfiguration *ui;

    QPixmap frame;
};

#endif // COLIBRICONFIGURATION_H
