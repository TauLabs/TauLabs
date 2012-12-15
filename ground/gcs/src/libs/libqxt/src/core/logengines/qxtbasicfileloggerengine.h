/****************************************************************************
 **
 ** Copyright (C) Qxt Foundation. Some rights reserved.
 **
 ** This file is part of the QxtCore module of the Qxt library.
 **
 ** This library is free software; you can redistribute it and/or modify it
 ** under the terms of the Common Public License, version 1.0, as published
 ** by IBM, and/or under the terms of the GNU Lesser General Public License,
 ** version 2.1, as published by the Free Software Foundation.
 **
 ** This file is provided "AS IS", without WARRANTIES OR CONDITIONS OF ANY
 ** KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT LIMITATION, ANY
 ** WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY OR
 ** FITNESS FOR A PARTICULAR PURPOSE.
 **
 ** You should have received a copy of the CPL and the LGPL along with this
 ** file. See the LICENSE file and the cpl1.0.txt/lgpl-2.1.txt files
 ** included with the source distribution for more information.
 ** If you did not receive a copy of the licenses, contact the Qxt Foundation.
 **
 ** <http://libqxt.org>  <foundation@libqxt.org>
 **
 ****************************************************************************/

#ifndef QXTBASICFILELOGGERENGINE_H
#define QXTBASICFILELOGGERENGINE_H

#include "qxtloggerengine.h"
#include "qxtabstractfileloggerengine.h"
#include <QTextStream>
#include <QFile>

/*******************************************************************************
    QBasicSTDLoggerEngine
    The basic logger engine included with QxtLogger.
*******************************************************************************/

class QxtBasicFileLoggerEnginePrivate;
class QXT_CORE_EXPORT QxtBasicFileLoggerEngine : public QxtAbstractFileLoggerEngine
{
public:
    QxtBasicFileLoggerEngine(const QString &fileName = QString());

public:
    QString dateFormat() const;
    void setDateFormat(const QString& format);

protected:
    virtual void writeToFile(const QString &level, const QVariantList &messages);

private:
    QXT_DECLARE_PRIVATE(QxtBasicFileLoggerEngine)
};

#endif // QXTBASICFILELOGGERENGINE_H
