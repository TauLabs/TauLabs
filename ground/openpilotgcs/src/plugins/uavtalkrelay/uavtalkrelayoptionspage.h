#ifndef UAVTALKRELAYOPTIONSPAGE_H
#define UAVTALKRELAYOPTIONSPAGE_H

#include "coreplugin/dialogs/ioptionspage.h"
#include <coreplugin/iconfigurableplugin.h>


class UavTalkRelayPlugin;
namespace Core {
    class IUAVGadgetConfiguration;
}

namespace Ui {
    class UavTalkRelayOptionsPage;
}

using namespace Core;

class UavTalkRelayOptionsPage : public IOptionsPage
{
Q_OBJECT
public:
    UavTalkRelayOptionsPage(QObject *parent = 0);
    virtual ~UavTalkRelayOptionsPage();

    QString id() const { return QLatin1String("settings"); }
    QString trName() const { return tr("settings"); }
    QString category() const { return "UAV Talk Relay";}
    QString trCategory() const { return "UAV Talk Relay"; }

    QWidget *createPage(QWidget *parent);
    void apply();
    void finish();

public slots:
private:
    Ui::UavTalkRelayOptionsPage *m_page;
    UavTalkRelayPlugin * m_config;
};

#endif // UAVTALKRELAYOPTIONSPAGE_H
