#ifndef ALARMSMONITORWIDGET_H
#define ALARMSMONITORWIDGET_H

#include <QObject>
#include <QGraphicsSvgItem>
#include <QGraphicsTextItem>
#include <QSvgRenderer>
#include <QTimer>

class AlarmsMonitorWidget : public QObject
{
    Q_OBJECT
public:
    static AlarmsMonitorWidget& getInstance()
    {
        static AlarmsMonitorWidget instance;
        return instance;
    }
    void init(QSvgRenderer * renderer, QGraphicsSvgItem *graph);
signals:
    
public slots:
    void processAlerts();
    void updateMessages();
private:
    AlarmsMonitorWidget();
    AlarmsMonitorWidget(AlarmsMonitorWidget const&);              // Don't Implement.
    void operator=(AlarmsMonitorWidget const&); // Don't implement
    QGraphicsSvgItem *error_sym;
    QGraphicsSvgItem *warning_sym;
    QGraphicsSvgItem *info_sym;
    QGraphicsTextItem *error_txt;
    QGraphicsTextItem *warning_txt;
    QGraphicsTextItem *info_txt;
    bool hasErrors;
    bool hasWarnings;
    bool hasInfos;
    QTimer alertTimer;
};

#endif // ALARMSMONITORWIDGET_H
