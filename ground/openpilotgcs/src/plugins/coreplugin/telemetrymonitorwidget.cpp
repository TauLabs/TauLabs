#include "telemetrymonitorwidget.h"

#include <QObject>
#include <QtGui>
#include <QtGui/QFont>
#include <QDebug>
#include <coreplugin/icore.h>
#include <globalmessaging.h>

#define DIMMED_SYMBOL 0.1

TelemetryMonitorWidget::TelemetryMonitorWidget(QWidget *parent) : QGraphicsView(parent),hasErrors(false),hasWarnings(false),hasInfos(false)
{
    setMinimumSize(180,100);
    setMaximumSize(180,100);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setAlignment(Qt::AlignCenter);
    setFrameStyle(QFrame::NoFrame);
    setStyleSheet("background:transparent;");
    setAttribute(Qt::WA_TranslucentBackground);
    setWindowFlags(Qt::FramelessWindowHint);

    QGraphicsScene *scene = new QGraphicsScene(0,0,180,100, this);

    QSvgRenderer *renderer = new QSvgRenderer();
    if (renderer->load(QString(":/core/images/tx-rx.svg"))) {
        graph = new QGraphicsSvgItem();
        graph->setSharedRenderer(renderer);
        graph->setElementId("txrxBackground");


        QString name;
        QGraphicsSvgItem* pt;

        error_sym=new QGraphicsSvgItem();
        error_sym->setSharedRenderer(renderer);
        error_sym->setElementId("error_sym");
        error_sym->setParentItem(graph);
        error_txt = new QGraphicsTextItem();
        error_txt->setDefaultTextColor(Qt::white);
        error_txt->setFont(QFont("Helvetica",22,2));
        error_txt->setParentItem(graph);
        error_txt->setPlainText("0");

        QRectF orig=renderer->boundsOnElement("error_sym");
        QMatrix Matrix = renderer->matrixForElement("error_sym");
        orig=Matrix.mapRect(orig);
        QTransform trans;
        trans.translate(orig.x(),orig.y());
        error_sym->setTransform(trans,false);
        trans.reset();
        int refY=orig.y();
        trans.translate(orig.x()+orig.width()-5,refY);
        error_txt->setTransform(trans,false);
        trans.reset();

        info_sym=new QGraphicsSvgItem();
        info_sym->setSharedRenderer(renderer);
        info_sym->setElementId("info_sym");
        info_sym->setParentItem(graph);
        info_txt = new QGraphicsTextItem();
        info_txt->setDefaultTextColor(Qt::white);
        info_txt->setFont(QFont("Helvetica",22,2));
        info_txt->setParentItem(graph);
        info_txt->setPlainText("0");
        orig=renderer->boundsOnElement("info_sym");
        Matrix = renderer->matrixForElement("info_sym");
        orig=Matrix.mapRect(orig);
        trans.translate(orig.x(),orig.y());
        info_sym->setTransform(trans,false);
        trans.reset();
        trans.translate(orig.x()-5+orig.width(),refY);
        info_txt->setTransform(trans,false);
        trans.reset();

        warning_sym=new QGraphicsSvgItem();
        warning_sym->setSharedRenderer(renderer);
        warning_sym->setElementId("warning_sym");
        warning_sym->setParentItem(graph);
        warning_txt = new QGraphicsTextItem();
        warning_txt->setDefaultTextColor(Qt::white);
        warning_txt->setFont(QFont("Helvetica",22,2));
        warning_txt->setParentItem(graph);
        warning_txt->setPlainText("0");
        orig=renderer->boundsOnElement("warning_sym");
        Matrix = renderer->matrixForElement("warning_sym");
        orig=Matrix.mapRect(orig);
        trans.translate(orig.x(),orig.y());
        warning_sym->setTransform(trans,false);
        trans.reset();
        trans.translate(orig.x()+orig.width()-20,refY);
        warning_txt->setTransform(trans,false);
        trans.reset();
        error_sym->setOpacity(0.1);
        warning_sym->setOpacity(0.1);
        info_sym->setOpacity(0.1);
        error_txt->setOpacity(0.1);
        warning_txt->setOpacity(0.1);
        info_txt->setOpacity(0.1);

        for (int i=0; i<NODE_NUMELEM; i++) {
            name = QString("tx%0").arg(i);
            if (renderer->elementExists(name)) {
                pt = new QGraphicsSvgItem();
                pt->setSharedRenderer(renderer);
                pt->setElementId(name);
                pt->setParentItem(graph);
                txNodes.append(pt);
            }

            name = QString("rx%0").arg(i);
            if (renderer->elementExists(name)) {
                pt = new QGraphicsSvgItem();
                pt->setSharedRenderer(renderer);
                pt->setElementId(name);
                pt->setParentItem(graph);
                rxNodes.append(pt);
            }
        }

        scene->addItem(graph);

        txSpeed = new QGraphicsTextItem();
        txSpeed->setDefaultTextColor(Qt::white);
        txSpeed->setFont(QFont("Helvetica",22,2));
        txSpeed->setParentItem(graph);
        scene->addItem(txSpeed);

        rxSpeed = new QGraphicsTextItem();
        rxSpeed->setDefaultTextColor(Qt::white);
        rxSpeed->setFont(QFont("Helvetica",22,2));
        rxSpeed->setParentItem(graph);
        scene->addItem(rxSpeed);

        scene->setSceneRect(graph->boundingRect());
        setScene(scene);
    }

    m_connected = false;
    txValue = 0.0;
    rxValue = 0.0;

    setMin(0.0);
    setMax(1200.0);

    showTelemetry();
    connect(&alertTimer,SIGNAL(timeout()),this,SLOT(processAlerts()));
    connect(Core::ICore::instance()->globalMessaging(),SIGNAL(newMessage(GlobalMessage*)),this,SLOT(updateMessages()));
    connect(Core::ICore::instance()->globalMessaging(),SIGNAL(deletedMessage()),this,SLOT(updateMessages()));
    connect(Core::ICore::instance()->globalMessaging(),SIGNAL(changedMessage(GlobalMessage*)),this,SLOT(updateMessages()));

    alertTimer.start(1000);
}

TelemetryMonitorWidget::~TelemetryMonitorWidget()
{
    while (!txNodes.isEmpty())
        delete txNodes.takeFirst();

    while (!rxNodes.isEmpty())
        delete rxNodes.takeFirst();
}

void TelemetryMonitorWidget::connected()
{
    m_connected = true;

    //flash the lights
    updateTelemetry(maxValue, maxValue);
}

void TelemetryMonitorWidget::disconnect()
{
    //flash the lights
    updateTelemetry(maxValue, maxValue);

    m_connected = false;
    updateTelemetry(0.0,0.0);
}
/*!
  \brief Called by the UAVObject which got updated

  Updates the numeric value and/or the icon if the dial wants this.
  */
void TelemetryMonitorWidget::updateTelemetry(double txRate, double rxRate)
{
    txValue = txRate;
    rxValue = rxRate;

    showTelemetry();
}

// Converts the value into an percentage:
// this enables smooth movement in moveIndex below
void TelemetryMonitorWidget::showTelemetry()
{
    txIndex = (txValue-minValue)/(maxValue-minValue) * NODE_NUMELEM;
    rxIndex = (rxValue-minValue)/(maxValue-minValue) * NODE_NUMELEM;

    if (m_connected)
        this->setToolTip(QString("Tx: %0 bytes/sec\nRx: %1 bytes/sec").arg(txValue).arg(rxValue));
    else
        this->setToolTip(QString("Disconnected"));

    int i;
    int nodeMargin = 8;
    int leftMargin = 60;
    QGraphicsItem* node;

    for (i=0; i < txNodes.count(); i++) {
        node = txNodes.at(i);
        node->setPos((i*(node->boundingRect().width() + nodeMargin)) + leftMargin, (node->boundingRect().height()/2) - 2);
        node->setVisible(m_connected && i < txIndex);
        node->update();
    }

    for (i=0; i < rxNodes.count(); i++) {
        node = rxNodes.at(i);
        node->setPos((i*(node->boundingRect().width() + nodeMargin)) + leftMargin, (node->boundingRect().height()*2) - 2);
        node->setVisible(m_connected && i < rxIndex);
        node->update();
    }

    QRectF rect = graph->boundingRect();
    txSpeed->setPos(rect.right() - 110, rect.top());
    txSpeed->setPlainText(QString("%0").arg(txValue));
    txSpeed->setVisible(m_connected);

    rxSpeed->setPos(rect.right() - 110, rect.top() + (rect.height() / 2));
    rxSpeed->setPlainText(QString("%0").arg(rxValue));
    rxSpeed->setVisible(m_connected);

    update();
}

void TelemetryMonitorWidget::showEvent(QShowEvent *event)
{
    Q_UNUSED(event);

    fitInView(graph, Qt::KeepAspectRatio);
}

void TelemetryMonitorWidget::resizeEvent(QResizeEvent* event)
{
    Q_UNUSED(event);

    graph->setPos(0,-130);
    fitInView(graph, Qt::KeepAspectRatio);
}

void TelemetryMonitorWidget::updateMessages()
{
    QString error;
    error.append("<html><head/><body>");
    foreach(Core::GlobalMessage * msg,Core::ICore::instance()->globalMessaging()->getActiveErrors())
    {
        QString temp;
        temp=QString("<p><span style=' font-size:11pt; font-weight:600;'>%0</span></p>").arg(msg->getBrief());
        error.append(temp);
        temp=QString("<p><span style=' font-style:italic;'>%0</span></p>").arg(msg->getDescription());
        error.append(temp);
    }
    error.append("</body></html>");
    QString warning;
    warning.append("<html><head/><body>");
    foreach(Core::GlobalMessage * msg,Core::ICore::instance()->globalMessaging()->getActiveWarnings())
    {
        QString temp;
        temp=QString("<p><span style=' font-size:11pt; font-weight:600;'>%0</span></p>").arg(msg->getBrief());
        warning.append(temp);
        temp=QString("<p><span style=' font-style:italic;'>%0</span></p>").arg(msg->getDescription());
        warning.append(temp);
    }
    warning.append("</body></html>");
    QString info;
    info.append("<html><head/><body>");
    foreach(Core::GlobalMessage * msg,Core::ICore::instance()->globalMessaging()->getActiveInfos())
    {
        QString temp;
        temp=QString("<p><span style=' font-size:11pt; font-weight:600;'>%0</span></p>").arg(msg->getBrief());
        info.append(temp);
        temp=QString("<p><span style=' font-style:italic;'>%0</span></p>").arg(msg->getDescription());
        info.append(temp);
    }
    info.append("</body></html>");
    error_txt->setPlainText(QString::number(Core::ICore::instance()->globalMessaging()->getActiveErrors().length()));
    if(Core::ICore::instance()->globalMessaging()->getActiveErrors().length()>0)
    {
        error_txt->setOpacity(1);
        hasErrors=true;
    }
    else
    {
        error="No errors";
        error_txt->setOpacity(DIMMED_SYMBOL);
        hasErrors=false;
    }
    warning_txt->setPlainText(QString::number(Core::ICore::instance()->globalMessaging()->getActiveWarnings().length()));
    if(Core::ICore::instance()->globalMessaging()->getActiveWarnings().length()>0)
    {
        warning_txt->setOpacity(1);
        hasWarnings=true;
    }
    else
    {
        warning="No warnings";
        warning_txt->setOpacity(DIMMED_SYMBOL);
        hasWarnings=false;
    }
    info_txt->setPlainText(QString::number(Core::ICore::instance()->globalMessaging()->getActiveInfos().length()));
    if(Core::ICore::instance()->globalMessaging()->getActiveInfos().length()>0)
    {
        info_txt->setOpacity(1);
        hasInfos=true;
    }
    else
    {
        info="No info";
        info_txt->setOpacity(DIMMED_SYMBOL);
        hasInfos=false;
    }
    error_sym->setToolTip(error);
    warning_sym->setToolTip(warning);
    info_sym->setToolTip(info);
}

void TelemetryMonitorWidget::processAlerts()
{
    static bool flag=true;
    flag = flag ^ true;
    if(hasErrors)
    {
        if(flag)
            error_sym->setOpacity(1);
        else
            error_sym->setOpacity(DIMMED_SYMBOL);
    }
    else
        error_sym->setOpacity(DIMMED_SYMBOL);
    if(hasWarnings)
    {
        if(flag)
            warning_sym->setOpacity(1);
        else
            warning_sym->setOpacity(DIMMED_SYMBOL);
    }
    else
        warning_sym->setOpacity(DIMMED_SYMBOL);
    if(hasInfos)
    {
        if(flag)
            info_sym->setOpacity(1);
        else
            info_sym->setOpacity(DIMMED_SYMBOL);
    }
    else
        info_sym->setOpacity(DIMMED_SYMBOL);
}

