/********************************************************************************
 *
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup OSGEarthViewPluging OSG Earth View plugin to visualize UAV in 3D
 * @{
 * @brief Osg Earth view of UAV
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

#ifndef OSGEARTHVIEWGADGETCONFIGURATION_H
#define OSGEARTHVIEWGADGETCONFIGURATION_H

#include <coreplugin/iuavgadgetconfiguration.h>

using namespace Core;

class OsgEarthviewGadgetConfiguration : public IUAVGadgetConfiguration
{
Q_OBJECT
public:
    explicit OsgEarthviewGadgetConfiguration(QString classId, QSettings* qSettings = 0, QObject *parent = 0);

    void saveConfig(QSettings* settings) const;
    IUAVGadgetConfiguration *clone();

private:
};

#endif // OSGEARTHVIEWGADGETCONFIGURATION_H
