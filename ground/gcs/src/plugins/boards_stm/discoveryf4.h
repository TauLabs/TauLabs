/**
 ******************************************************************************
 *
 * @file       discoveryf4.h
 * @author     Tau Labs, http://github.com/TauLabs, Copyright (C) 2013
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_Stm Stm boards support Plugin
 * @{
 * @brief Plugin to support boards from STM
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
#ifndef DISCOVERYF4_H
#define DISCOVERYF4_H

#include <coreplugin/iboardtype.h>

class IBoardType;

class DiscoveryF4 : public Core::IBoardType
{
public:
    DiscoveryF4();
    virtual ~DiscoveryF4();

    virtual QString shortName();
    virtual QString boardDescription();
    virtual bool queryCapabilities(BoardCapabilities capability);
    virtual QStringList getSupportedProtocols();
    virtual QPixmap getBoardPicture();
    virtual QString getHwUAVO();
};


#endif // DISCOVERYF4_H
