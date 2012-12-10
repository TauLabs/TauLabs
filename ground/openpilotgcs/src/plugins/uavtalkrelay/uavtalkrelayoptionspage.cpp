/**
 ******************************************************************************
 * @file       uavtalkrelayoptionspage.c
 * @author     The PhoenixPilot Team, http://github.com/PhoenixPilot
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVTalk relay plugin
 * @{
 *
 * @brief Relays UAVTalk data trough UDP to another GCS
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
#include "uavtalkrelayplugin.h"
#include "uavtalkrelayoptionspage.h"
#include <QtGui/QLabel>
#include <QtGui/QComboBox>
#include <QtGui/QSpinBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QRadioButton>
#include <QtGui/QHBoxLayout>
#include <QtGui/QVBoxLayout>
#include "ui_uavtalkrelayoptionspage.h"

UavTalkRelayOptionsPage::UavTalkRelayOptionsPage(QObject *parent) :
    IOptionsPage(parent)
{
    m_config=qobject_cast<UavTalkRelayPlugin *>(parent);
    connect(this,SIGNAL(settingsUpdated()),m_config,SLOT(updateSettings()));
}
UavTalkRelayOptionsPage::~UavTalkRelayOptionsPage()
{
}
QWidget *UavTalkRelayOptionsPage::createPage(QWidget *parent)
{

    m_page = new Ui::UavTalkRelayOptionsPage();
    QWidget *w = new QWidget(parent);
    m_page->setupUi(w);
    connect(m_page->pbAddRule,SIGNAL(clicked()),this,SLOT(addRule()));
    connect(m_page->pbDeleteRule,SIGNAL(clicked()),this,SLOT(deleteRule()));
    m_page->cbAddRuleAccessType->addItem("Read Only",UavTalkRelayComon::ReadOnly);
    m_page->cbAddRuleAccessType->addItem("Write Only",UavTalkRelayComon::WriteOnly);
    m_page->cbAddRuleAccessType->addItem("Read and Write",UavTalkRelayComon::ReadWrite);
    m_page->cbAddRuleAccessType->addItem("none",UavTalkRelayComon::None);
    m_page->cbDefaultAccess->addItem("Read Only",UavTalkRelayComon::ReadOnly);
    m_page->cbDefaultAccess->addItem("Write Only",UavTalkRelayComon::WriteOnly);
    m_page->cbDefaultAccess->addItem("Read and Write",UavTalkRelayComon::ReadWrite);
    m_page->cbDefaultAccess->addItem("none",UavTalkRelayComon::None);
    m_page->cbDefaultAccess->setCurrentIndex(m_page->cbDefaultAccess->findData(m_config->m_DefaultRule));
    ExtensionSystem::PluginManager* pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager * objMngr = pm->getObject<UAVObjectManager>();
    QList < QList <UAVObject *> > objList=objMngr->getObjects();
    foreach(QList <UAVObject *> obj,objList)
    {
        m_page->cbAddRuleUAVO->addItem(obj[0]->getName(),obj[0]->getObjID());
    }

    m_page->ListeningPort->setValue(m_config->m_Port);
    m_page->ListeningInterface->setText(m_config->m_IpAddress);
    foreach(QString host,m_config->rules.keys())
    {
        foreach(quint32 uavo,m_config->rules.value(host).keys())
        {
            int i=m_page->twRules->rowCount();
            m_page->twRules->setRowCount(i+1);
            QTableWidgetItem * newItem=new QTableWidgetItem(tr("%1").arg(host));
            m_page->twRules->setItem(i,0,newItem);
            QString uavtxt=m_page->cbAddRuleUAVO->itemText(m_page->cbAddRuleUAVO->findData(uavo));
            QString aTypetxt=m_page->cbAddRuleAccessType->itemText(m_page->cbAddRuleAccessType->findData(m_config->rules.value(host).value(uavo)));
            newItem=new QTableWidgetItem(tr("%1").arg(uavtxt));
            m_page->twRules->setItem(i,1,newItem);
            newItem=new QTableWidgetItem(tr("%1").arg(aTypetxt));
            m_page->twRules->setItem(i,2,newItem);
        }
    }
    m_page->twRules->setSelectionBehavior(QAbstractItemView::SelectRows);
    return w;
}

void UavTalkRelayOptionsPage::apply()
{
    m_config->m_Port=m_page->ListeningPort->value();
    m_config->m_IpAddress=m_page->ListeningInterface->text();
    m_config->rules.clear();
    for(int i=0;i < m_page->twRules->rowCount();++i)
    {
        QString host=m_page->twRules->item(i,0)->text();
        quint32 uavo=m_page->cbAddRuleUAVO->itemData(m_page->cbAddRuleUAVO->findText(m_page->twRules->item(i,1)->text())).toInt();
        UavTalkRelayComon::accessType aType=(UavTalkRelayComon::accessType)m_page->cbAddRuleAccessType->itemData(m_page->cbAddRuleAccessType->findText(m_page->twRules->item(i,2)->text())).toInt();
        if(m_config->rules.keys().contains(host))
        {
           QHash <quint32,UavTalkRelayComon::accessType> temp=m_config->rules.value(host);
           temp.insert(uavo,aType);
        }
        else
        {
            QHash <quint32,UavTalkRelayComon::accessType> temp;
            temp.insert(uavo,aType);
            m_config->rules.insert(host,temp);
        }
    }
    emit settingsUpdated();
}

void UavTalkRelayOptionsPage::finish()
{
    delete m_page;
}

void UavTalkRelayOptionsPage::addRule()
{
    int i=m_page->twRules->rowCount();
    m_page->twRules->setRowCount(i+1);
    QTableWidgetItem * newItem=new QTableWidgetItem(tr("%1").arg(m_page->leAddRuleHost->text()));
    m_page->twRules->setItem(i,0,newItem);
    newItem=new QTableWidgetItem(tr("%1").arg(m_page->cbAddRuleUAVO->currentText()));
    m_page->twRules->setItem(i,1,newItem);
    newItem=new QTableWidgetItem(tr("%1").arg(m_page->cbAddRuleAccessType->currentText()));
    m_page->twRules->setItem(i,2,newItem);
}

void UavTalkRelayOptionsPage::deleteRule()
{
    m_page->twRules->removeRow(m_page->twRules->currentRow());
}
