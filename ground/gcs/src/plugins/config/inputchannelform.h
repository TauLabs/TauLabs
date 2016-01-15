#ifndef INPUTCHANNELFORM_H
#define INPUTCHANNELFORM_H

#include <QWidget>
#include "configinputwidget.h"
namespace Ui {
    class inputChannelForm;
}

class inputChannelForm : public ConfigTaskWidget
{
    Q_OBJECT

public:
    typedef enum { CHANNELFUNC_RC, CHANNELFUNC_RSSI } ChannelFunc;

    explicit inputChannelForm(QWidget *parent = 0, bool showlegend=false, bool showSlider=true, ChannelFunc chanType = CHANNELFUNC_RC);
    ~inputChannelForm();
    friend class ConfigInputWidget;
    void setName(QString &name);
private slots:
    void minMaxUpdated();
    void groupUpdated();
    void channelDropdownUpdated(int);
    void channelNumberUpdated(int);

private:
    Ui::inputChannelForm *ui;
    ChannelFunc m_chanType;
};

#endif // INPUTCHANNELFORM_H
