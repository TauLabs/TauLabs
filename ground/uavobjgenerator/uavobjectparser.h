/**
 ******************************************************************************
 *
 * @file       uavobjectparser.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      Parses XML files and extracts object information.
 *
 * @see        The GNU Public License (GPL) Version 3
 *
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

#ifndef UAVOBJECTPARSER_H
#define UAVOBJECTPARSER_H

#include <QString>
#include <QStringList>
#include <QList>
#include <QSet>
#include <QDomDocument>
#include <QDomElement>
#include <QDomNode>
#include <QByteArray>

/**
 * The maximum size of UAVOs is limited by the FlashFS filesystem in the flight code
 * The flash slot size is 256 bytes which is comprised of the FlashFS header (12 bytes)
 * and the UAVO. This leaves a maximum of 244 bytes for the UAVO.
 */
#define UAVO_MAX_SIZE 244

// Types
typedef enum {
    FIELDTYPE_INT8 = 0,
    FIELDTYPE_INT16,
    FIELDTYPE_INT32,
    FIELDTYPE_UINT8,
    FIELDTYPE_UINT16,
    FIELDTYPE_UINT32,
    FIELDTYPE_FLOAT32,
    FIELDTYPE_ENUM
} FieldType;

typedef struct FieldInfo_s FieldInfo;
typedef struct ObjectInfo_s ObjectInfo;

struct FieldInfo_s {
    QString name;
    QString units;
    FieldType type;
    int numElements;
    int numBytes;
    QStringList elementNames;
    QStringList options; // for enums only
    QString parentName;  // optional, for enums only
    bool defaultElementNames;
    QStringList defaultValues;
    QString limitValues;
    QString description;

    FieldInfo *parent;
    ObjectInfo *parentObj;
};

/**
 * Object update mode
 */
typedef enum {
    UPDATEMODE_MANUAL =    0, /** Manually update object, by calling the updated() function */
    UPDATEMODE_PERIODIC =  1, /** Automatically update object at periodic intervals */
    UPDATEMODE_ONCHANGE  = 2, /** Only update object when its data changes */
    UPDATEMODE_THROTTLED = 3  /** Object is updated on change, but not more often than the interval time */
} UpdateMode;


typedef enum {
    ACCESS_READWRITE = 0,
    ACCESS_READONLY = 1
} AccessMode;

struct ObjectInfo_s {
    QString name;
    QString namelc; /** name in lowercase */
    QString filename;
    quint32 id;
    bool isSingleInst;
    bool isSettings;
    AccessMode gcsAccess;
    AccessMode flightAccess;
    bool flightTelemetryAcked;
    UpdateMode flightTelemetryUpdateMode; /** Update mode used by the autopilot (UpdateMode) */
    int flightTelemetryUpdatePeriod; /** Update period used by the autopilot (only if telemetry mode is PERIODIC) */
    bool gcsTelemetryAcked;
    UpdateMode gcsTelemetryUpdateMode; /** Update mode used by the GCS (UpdateMode) */
    int gcsTelemetryUpdatePeriod; /** Update period used by the GCS (only if telemetry mode is PERIODIC) */
    UpdateMode loggingUpdateMode; /** Update mode used by the logging module (UpdateMode) */
    int loggingUpdatePeriod; /** Update period used by the logging module (only if logging mode is PERIODIC) */
    QList<FieldInfo*> fields; /** The data fields for the object **/
    QString description; /** Description used for Doxygen **/
    QString category; /** Description used for Doxygen **/
    int numBytes;
    QSet<ObjectInfo*> parents;
};

class UAVObjectParser
{
public:

    // Functions
    UAVObjectParser();
    QString parseXML(QString& xml, QString& filename);
    QString resolveParents();
    void calculateAllIds();
    int getNumObjects();
    QList<ObjectInfo*> getObjectInfo();
    QString getObjectName(int objIndex);
    quint32 getObjectID(int objIndex);

    ObjectInfo* getObjectByIndex(int objIndex);
    ObjectInfo* getObjectByName(QString& name);
    FieldInfo* getFieldByName(QString &name, ObjectInfo **objRet);
    int findOptionIndex(FieldInfo *field, quint32 inputIdx);

    int getNumBytes(int objIndex);

    quint64 getUavoHash();

    QStringList all_units;

private:
    QList<ObjectInfo*> objInfo;
    QStringList fieldTypeStrXML;
    QList<int> fieldTypeNumBytes;
    QStringList updateModeStr;
    QStringList updateModeStrXML;
    QStringList accessModeStr;
    QStringList accessModeStrXML;
    quint64 uavoHash;

    QString genErrorMsg(QString& fileName, QString errMsg, int errorLine, int errorCol);

    QString processObjectAttributes(QDomNode& node, ObjectInfo* info);
    QString processObjectFields(QDomNode& childNode, ObjectInfo* info);
    QString processObjectAccess(QDomNode& childNode, ObjectInfo* info);
    QString processObjectDescription(QDomNode& childNode, QString * description);
    QString processObjectCategory(QDomNode& childNode, QString * category);
    QString processObjectMetadata(QDomNode& childNode, UpdateMode* mode, int* period, bool* acked);
    void calculateID(ObjectInfo* info);
    void calculateSize(ObjectInfo* info);
    quint32 updateHash(quint32 value, quint32 hash);
    quint32 updateHash(QString& value, quint32 hash);
    int resolveFieldParent(ObjectInfo *item, FieldInfo *field);
    int checkDefaultValues(FieldInfo *field);
};

#endif // UAVOBJECTPARSER_H
