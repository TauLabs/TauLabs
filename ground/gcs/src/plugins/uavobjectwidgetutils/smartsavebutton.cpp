/**
 ******************************************************************************
 *
 * @file       smartsavebutton.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVObjectWidgetUtils Plugin
 * @{
 * @brief Utility plugin for UAVObject to Widget relation management
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
#include "smartsavebutton.h"

smartSaveButton::smartSaveButton()
{

}

/**
 * @brief smartSaveButton::addButtons
 * Called only by the ConfigTaskWidget when adding the smart save buttons, depending
 * on whether we want Apply, Save, or Apply & Save.
 * @param save
 * @param apply
 */
void smartSaveButton::addButtons(QPushButton *save, QPushButton *apply)
{
    buttonList.insert(save,save_button);
    buttonList.insert(apply,apply_button);
    connect(save,SIGNAL(clicked()),this,SLOT(processClick()));
    connect(apply,SIGNAL(clicked()),this,SLOT(processClick()));
}

/**
 * @brief smartSaveButton::addApplyButton
 * Called only by the ConfigTaskWidget when adding the smart save buttons, depending
 * on whether we want Apply, Save, or Apply & Save.
 * @param apply
 */
void smartSaveButton::addApplyButton(QPushButton *apply)
{
    buttonList.insert(apply,apply_button);
    connect(apply,SIGNAL(clicked()),this,SLOT(processClick()));
}

/**
 * @brief smartSaveButton::addSaveButton
 * Called only by the ConfigTaskWidget when adding the smart save buttons, depending
 * on whether we want Apply, Save, or Apply & Save.
 * @param save
 */
void smartSaveButton::addSaveButton(QPushButton *save)
{
    buttonList.insert(save,save_button);
    connect(save,SIGNAL(clicked()),this,SLOT(processClick()));
}

/**
 * @brief smartSaveButton::processClick
 */
void smartSaveButton::processClick()
{
    emit beginOp();
    bool save=false;
    QPushButton *button=qobject_cast<QPushButton *>(sender());
    if(!button)
        return;
    if(buttonList.value(button)==save_button)
        save=true;
    processOperation(button,save);

}

/**
 * @brief smartSaveButton::processOperation
 * This is where actual operation processing takes place.
 * @param button
 * @param save is true if we want to issue a saveObjectToFlash request after sending it to the
 *             remote side
 */
void smartSaveButton::processOperation(QPushButton * button,bool save)
{
    emit preProcessOperations();
    if(button)
    {
        button->setEnabled(false);
        button->setIcon(QIcon(":/uploader/images/system-run.svg"));
    }
    QTimer timer;
    timer.setSingleShot(true);
    bool error=false;
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectUtilManager* utilMngr = pm->getObject<UAVObjectUtilManager>();
    foreach(UAVDataObject * obj,objects)
    {
        if(!obj->getIsPresentOnHardware())
            continue;
        UAVObject::Metadata mdata= obj->getMetadata();
        if(UAVObject::GetGcsAccess(mdata)==UAVObject::ACCESS_READONLY)
            continue;
        upload_result = false;
        current_object = obj;

        // We only allow to send/save Settings objects using this method
        if (!obj->isSettings()) {
            qDebug() << "[smartsavebutton.cpp] Error, tried to apply/save a non-settings object";
            continue;
        }

        qDebug() << "[smartsavebutton.cpp] Sending object to remote end - " << obj->getName();
        connect(obj,SIGNAL(transactionCompleted(UAVObject*,bool)),this,SLOT(transaction_finished(UAVObject*, bool)));
        connect(&timer,SIGNAL(timeout()),&loop,SLOT(quit()));
        obj->updated();
        // Three things can happen now:
        // - If object is ACK'ed then we'll get a transactionCompleted signal once ACK arrives
        //   with a success value at 'true'
        // - If remote side does not know it, or object is not ACK'ed then we'll get a timeout, which we catch below
        // - If object is ACK'ed and message gets lost, we'll get a transactionCompleted with a
        //   success value at 'false'
        //
        // Note: settings objects should always be ACK'ed, so a timeout should always be because of a lost
        //       message over the telemetry link.
        //
        // Note 2: the telemetry link does max 3 tries with 250ms interval when sending an object
        //         update over the link
        timer.start(1000);
        loop.exec();
        if (!timer.isActive())
            qDebug() << "[smartsavebutton.cpp] Upload timeout for object" << obj->getName();
        timer.stop();
        disconnect(obj,SIGNAL(transactionCompleted(UAVObject*,bool)),this,SLOT(transaction_finished(UAVObject*, bool)));
        disconnect(&timer,SIGNAL(timeout()),&loop,SLOT(quit()));

        if(upload_result==false) {
            qDebug() << "[smartsavebutton.cpp] Object upload error:" << obj->getName();
            error = true;
            continue;
        }

        // Now object is uploaded, we can move on to saving it to flash:
        save_result=false;
        current_objectID=obj->getObjID();

        if(save && (obj->isSettings()))
        {
            qDebug() << "[smartsavebutton.cpp] Save request for object" << obj->getName();
            connect(utilMngr,SIGNAL(saveCompleted(int,bool)),this,SLOT(saving_finished(int,bool)));
            connect(&timer,SIGNAL(timeout()),&loop,SLOT(quit()));
            utilMngr->saveObjectToFlash(obj);
            // Now, here is what will happen:
            // - saveObjectToFlash is going to attempt the save operation
            // - If save succeeds, then it will issue a saveCompleted signal with the ObjectID and 'true'
            // - If save fails, either because of error or timeout, then it will issue a saveCompleted signal
            //   with the ObjectID and 'false'.
            //
            // Note: in case of link timeout, the telemetry layer will retry up to 2 times, we don't
            // need to retry ourselves here.
            //
            // Note 2: saveObjectToFlash manages save operations in a queue, so there is no guarantee that
            // the first "saveCompleted" signal is for the object we just asked to save.
            timer.start(2000);
            loop.exec();
            if(!timer.isActive())
                qDebug() << "[smartsavebutton.cpp] Saving timeout for object" << obj->getName();
            timer.stop();
            disconnect(utilMngr,SIGNAL(saveCompleted(int,bool)),this,SLOT(saving_finished(int,bool)));
            disconnect(&timer,SIGNAL(timeout()),&loop,SLOT(quit()));

            if( save_result == false)
            {
                qDebug() << "[smartsavebutton.cpp] failed to save:" << obj->getName();
                error = true;
            }
        }
    }
    if(button)
        button->setEnabled(true);
    if(!error)
    {
        if(button)
            button->setIcon(QIcon(":/uploader/images/dialog-apply.svg"));
        emit saveSuccessfull();
    }
    else
    {
        if(button)
            button->setIcon(QIcon(":/uploader/images/process-stop.svg"));
    }
    emit endOp();
}

/**
 * @brief smartSaveButton::setObjects
 * Sets all monitored objects in one operation
 * @param list
 */
void smartSaveButton::setObjects(QList<UAVDataObject *> list)
{
    objects=list;
}

/**
 * @brief smartSaveButton::addObject
 * The smartSaveButton contains a list of objects it will work with, addObject
 * is used to add a new object to a smartSaveButton instance.
 * @param obj
 */
void smartSaveButton::addObject(UAVDataObject * obj)
{
    Q_ASSERT(obj);
    if(!objects.contains(obj))
        objects.append(obj);
}

/**
 * @brief smartSaveButton::removeObject
 * The smartSaveButton contains a list of objects it will work with, addObject
 * is used to remove an object from a smartSaveButton instance.
 * @param obj
 */
void smartSaveButton::removeObject(UAVDataObject * obj)
{
    if(objects.contains(obj))
        objects.removeAll(obj);
}

/**
 * @brief smartSaveButton::removeAllObjects
 * Remove all tracked objects at once.
 */
void smartSaveButton::removeAllObjects()
{
    objects.clear();
}

void smartSaveButton::clearObjects()
{
    objects.clear();
}

void smartSaveButton::transaction_finished(UAVObject* obj, bool result)
{
    if(current_object==obj)
    {
        upload_result=result;
        loop.quit();
    }
}

void smartSaveButton::saving_finished(int id, bool result)
{
    // The saveOjectToFlash method manages its own save queue, so we can be
    // in a situation where we get a saving_finished message for an object
    // which is not the one we're interested in, hence the check below:
    if((quint32)id==current_objectID)
    {
        save_result=result;
        loop.quit();
    }
}

void smartSaveButton::enableControls(bool value)
{
    foreach(QPushButton * button,buttonList.keys())
        button->setEnabled(value);
}

void smartSaveButton::resetIcons()
{
    foreach(QPushButton * button,buttonList.keys())
        button->setIcon(QIcon());
}

void smartSaveButton::apply()
{
    processOperation(NULL,false);
}

void smartSaveButton::save()
{
    processOperation(NULL,true);
}


