/**
 ******************************************************************************
 * @file       pathplannergadget.h
 * @author     PhoenixPilot Project, http://github.com/PhoenixPilot Copyright (C) 2012.
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup Path Planner GCS Plugins
 * @{
 * @addtogroup PathPlannerGadgetPlugin Path Planner Gadget Plugin
 * @{
 * @brief A gadget to edit a list of waypoints
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

#ifndef PathPlannerGADGET_H_
#define PathPlannerGADGET_H_

#include <coreplugin/iuavgadget.h>

namespace Core {
class IUAVGadget;
}

class PathPlannerGadgetWidget;

using namespace Core;

class PathPlannerGadget : public Core::IUAVGadget
{
    Q_OBJECT
public:
    PathPlannerGadget(QString classId, PathPlannerGadgetWidget *widget, QWidget *parent = 0);
    ~PathPlannerGadget();

    QList<int> context() const { return m_context; }
    QWidget *widget() { return m_widget; }
    QString contextHelpId() const { return QString(); }

    void loadConfiguration(IUAVGadgetConfiguration* config);
private:
    QWidget *m_widget;
	QList<int> m_context;
};


#endif // PathPlannerGADGET_H_
