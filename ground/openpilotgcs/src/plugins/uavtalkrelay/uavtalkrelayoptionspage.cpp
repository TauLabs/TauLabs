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
}
UavTalkRelayOptionsPage::~UavTalkRelayOptionsPage()
{
}
QWidget *UavTalkRelayOptionsPage::createPage(QWidget *parent)
{

    m_page = new Ui::UavTalkRelayOptionsPage();
    QWidget *w = new QWidget(parent);
    m_page->setupUi(w);

    m_page->Port->setValue(m_config->Port());
    m_page->IpAdress->setText(m_config->IpAdress());
    m_page->UseTCP->setChecked(m_config->UseTCP()?true:false);
    m_page->UseUDP->setChecked(m_config->UseTCP()?false:true);

    return w;
}

void UavTalkRelayOptionsPage::apply()
{
    m_config->setPort(m_page->Port->value());
    m_config->setIpAdress(m_page->IpAdress->text());
    m_config->setUseTCP(m_page->UseTCP->isChecked()?1:0);
}

void UavTalkRelayOptionsPage::finish()
{
    delete m_page;
}
