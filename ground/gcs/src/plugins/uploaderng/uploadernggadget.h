/**
 ******************************************************************************
 *
 * @file       uploadernggadget.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup  Uploaderng Uploaderng Plugin
 * @{
 * @brief The Tau Labs uploader plugin gadget
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

#ifndef UPLOADERNGGADGET_H
#define UPLOADERNGGADGET_H

#include <coreplugin/iuavgadget.h>
#include "uploadernggadgetwidget.h"
#include "uploaderng_global.h"

class IUAVGadget;
class QWidget;
class QString;
class UploaderngGadgetWidget;

using namespace Core;

namespace uploaderng {

class UPLOADERNG_EXPORT UploaderngGadget : public Core::IUAVGadget
{
    Q_OBJECT
public:
    UploaderngGadget(QString classId, UploaderngGadgetWidget *widget, QWidget *parent = 0);
    ~UploaderngGadget();

    QWidget *widget() { return m_widget; }
    void loadConfiguration(IUAVGadgetConfiguration* config);

private:
    UploaderngGadgetWidget *m_widget;
};
}
#endif // UPLOADERNGGADGET_H

