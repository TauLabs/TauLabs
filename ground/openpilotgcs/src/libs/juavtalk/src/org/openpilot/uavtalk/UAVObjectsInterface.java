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

package org.abovegroundlabs.uavtalk;

/**
 ******************************************************************************
 *
 * @file       UAVObjectsInterface.java 
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @brief      a Interface to a UAVObjects collection
 *
 ****************************************************************************
*/
public interface UAVObjectsInterface {

    public void init();
    public UAVObject[] getUAVObjectArray();
    public UAVObject getObjectByID(int id) ;
     
    public UAVObject getObjectByName(String name);
    public void printAll();
    
}
