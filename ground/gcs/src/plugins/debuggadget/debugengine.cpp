#include "debugengine.h"

debugengine::debugengine()
{
}

debugengine *debugengine::getInstance()
{
    static debugengine objectInstance;
    return &objectInstance;
}

void debugengine::writeWarning(const QString &message)
{
    emit warning(message);
}

void debugengine::writeDebug(const QString &message)
{
    emit debug(message);
}

void debugengine::writeCritical(const QString &message)
{
    emit critical(message);
}

void debugengine::writeFatal(const QString &message)
{
    emit fatal(message);
}
