/**
 ******************************************************************************
 *
 * @file       mocapgadget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup MoCapPlugin Motion Capture Plugin
 * @{
 * @brief The Hardware In The Loop plugin 
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

#ifndef MOCAP_H
#define MOCAP_H

#include <coreplugin/iuavgadget.h>
#include "mocapwidget.h"

class IUAVGadget;
class QWidget;
class QString;

using namespace Core;

class MoCapGadget : public Core::IUAVGadget
{
    Q_OBJECT
public:
    MoCapGadget(QString classId, MoCapWidget *widget, QWidget *parent = 0);
    ~MoCapGadget();

    QWidget *widget() { return m_widget; }
    void loadConfiguration(IUAVGadgetConfiguration* config);

signals:
	void changeConfiguration();

private:
    MoCapWidget* m_widget;
};


#endif // MOCAP_H
