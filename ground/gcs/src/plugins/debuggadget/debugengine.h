#ifndef DEBUGENGINE_H
#define DEBUGENGINE_H
#include <QTextBrowser>
#include <QPointer>
#include <QMutex>
#include <QObject>


class debugengine : public QObject {
    Q_OBJECT
    // Add all missing constructor etc... to have singleton
    debugengine();
public:
    static debugengine *getInstance();
    void writeWarning(const QString &message);
    void writeDebug(const QString &message);
    void writeCritical(const QString &message);
    void writeFatal(const QString &message);

signals:
    void warning(QString);
    void debug(QString);
    void critical(QString);
    void fatal(QString);

};

#endif // DEBUGENGINE_H
