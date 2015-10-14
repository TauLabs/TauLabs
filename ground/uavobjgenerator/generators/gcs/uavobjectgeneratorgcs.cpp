/**
 ******************************************************************************
 *
 * @file       uavobjectgeneratorgcs.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      produce gcs code for uavobjects
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

#include "uavobjectgeneratorgcs.h"
using namespace std;

bool UAVObjectGeneratorGCS::generate(UAVObjectParser* parser,QString templatepath,QString outputpath) {

    fieldTypeStrCPP << "qint8" << "qint16" << "qint32" <<
        "quint8" << "quint16" << "quint32" << "float" << "quint8";

    fieldTypeStrCPPClass << "INT8" << "INT16" << "INT32"
        << "UINT8" << "UINT16" << "UINT32" << "FLOAT32" << "ENUM";

    gcsCodePath = QDir( templatepath + QString(GCS_CODE_DIR));
    gcsOutputPath = QDir( outputpath + QString("gcs") );
    gcsOutputPath.mkpath(gcsOutputPath.absolutePath());

    gcsCodeTemplate = readFile( gcsCodePath.absoluteFilePath("uavobjecttemplate.cpp") );
    gcsIncludeTemplate = readFile( gcsCodePath.absoluteFilePath("uavobjecttemplate.h") );
    QString gcsInitTemplate = readFile( gcsCodePath.absoluteFilePath("uavobjectsinittemplate.cpp") );
    QString gcsVersionTemplate = readFile( gcsCodePath.absoluteFilePath("uavogcsversiontemplate.h") );

    if (gcsCodeTemplate.isEmpty() || gcsIncludeTemplate.isEmpty() || gcsInitTemplate.isEmpty() || gcsVersionTemplate.isEmpty()) {
        std::cerr << "Problem reading gcs code templates" << endl;
        return false;
    }

    QString objInc;
    QString gcsObjInit;

    for (int objidx = 0; objidx < parser->getNumObjects(); ++objidx) {
        ObjectInfo* info=parser->getObjectByIndex(objidx);
        process_object(info);

        gcsObjInit.append("    objMngr->registerObject( new " + info->name + "() );\n");
        objInc.append("#include \"" + info->namelc + ".h\"\n");
    }

    // Write the gcs object inialization files
    gcsInitTemplate.replace( QString("$(OBJINC)"), objInc);
    gcsInitTemplate.replace( QString("$(OBJINIT)"), gcsObjInit);
    bool res = writeFileIfDiffrent( gcsOutputPath.absolutePath() + "/uavobjectsinit.cpp", gcsInitTemplate );
    if (!res) {
        cout << "Error: Could not write output files" << endl;
        return false;
    }

    gcsVersionTemplate.replace( QString("$(UAVOHASH)"), 
                QString("0x%1").arg(parser->getUavoHash(), 16, 16, QChar('0')));
    res = writeFileIfDiffrent( gcsOutputPath.absolutePath() + "/uavogcsversion.h", gcsVersionTemplate );
    if (!res) {
        cout << "Error: Could not write output files" << endl;
        return false;
    }

    return true; // if we come here everything should be fine
}

QString UAVObjectGeneratorGCS::form_enum_name(const QString& objectName,
        const QString& fieldName, const QString& option) {
    QString s = "%3::%1_%2";

    if (objectName.isEmpty()) {
        s = "%1_%2";

        return s.arg( fieldName.toUpper() )
            .arg( option.toUpper().replace(QRegExp(ENUM_SPECIAL_CHARS), ""));
    }

    return s.arg( fieldName.toUpper() )
        .arg( option.toUpper().replace(QRegExp(ENUM_SPECIAL_CHARS), ""))
        .arg( objectName);
}

/**
 * Generate the GCS object files
 */
bool UAVObjectGeneratorGCS::process_object(ObjectInfo* info)
{
    if (info == NULL)
        return false;

    // Prepare output strings
    QString outInclude = gcsIncludeTemplate;
    QString outCode = gcsCodeTemplate;

    // Replace common tags
    replaceCommonTags(outInclude, info);
    replaceCommonTags(outCode, info);

    // Replace the $(PARENT_INCLUDES) tag
    QString parentIncludes;

    foreach (ObjectInfo *parent, info->parents) {
        parentIncludes.append( QString("#include <%1.h>\r\n").arg(parent->namelc));
    }

    outInclude.replace(QString("$(PARENT_INCLUDES)"), parentIncludes);

    // Replace the $(DATAFIELDS) tag
    QString type;
    QString fields;
    for (int n = 0; n < info->fields.length(); ++n)
    {
        // Determine type
        type = fieldTypeStrCPP[info->fields[n]->type];
        // Append field
        if ( info->fields[n]->numElements > 1 )
        {
            fields.append( QString("        %1 %2[%3];\n").arg(type).arg(info->fields[n]->name)
                           .arg(info->fields[n]->numElements) );
        }
        else
        {
            fields.append( QString("        %1 %2;\n").arg(type).arg(info->fields[n]->name) );
        }
    }
    outInclude.replace(QString("$(DATAFIELDS)"), fields);

    // Replace $(PROPERTIES) and related tags
    QString properties;
    QString propertiesImpl;
    QString propertyGetters;
    QString propertySetters;
    QString propertyNotifications;
    QString propertyNotificationsImpl;

    //to avoid name conflicts
    QStringList reservedProperties;
    reservedProperties << "Description" << "Metadata";

    for (int n = 0; n < info->fields.length(); ++n)
    {
        FieldInfo *field = info->fields[n];

        if (reservedProperties.contains(field->name))
            continue;

        // Determine type
        type = fieldTypeStrCPP[field->type];
        // Append field
        if ( field->numElements > 1 ) {
            //add both field(elementIndex)/setField(elemntIndex,value) and field_element properties
            //field_element is more convenient if only certain element is used
            //and much easier to use from the qml side
            propertyGetters +=
                    QString("    Q_INVOKABLE %1 get%2(quint32 index) const;\n")
                    .arg(type).arg(field->name);
            propertiesImpl +=
                    QString("%1 %2::get%3(quint32 index) const\n"
                            "{\n"
                            "   QMutexLocker locker(mutex);\n"
                            "   return data.%3[index];\n"
                            "}\n")
                    .arg(type).arg(info->name).arg(field->name);
            propertySetters +=
                    QString("    void set%1(quint32 index, %2 value);\n")
                    .arg(field->name).arg(type);
            propertiesImpl +=
                    QString("void %1::set%2(quint32 index, %3 value)\n"
                            "{\n"
                            "   mutex->lock();\n"
                            "   bool changed = data.%2[index] != value;\n"
                            "   data.%2[index] = value;\n"
                            "   mutex->unlock();\n"
                            "   if (changed) emit %2Changed(index,value);\n"
                            "}\n\n")
                    .arg(info->name).arg(field->name).arg(type);
            propertyNotifications +=
                    QString("    void %1Changed(quint32 index, %2 value);\n")
                    .arg(field->name).arg(type);

            for (int elementIndex = 0; elementIndex < field->numElements; elementIndex++) {
                QString elementName = field->elementNames[elementIndex];
                properties += QString("    Q_PROPERTY(%1 %2 READ get%2 WRITE set%2 NOTIFY %2Changed);\n")
                        .arg(type).arg(field->name+"_"+elementName);
                propertyGetters +=
                        QString("    Q_INVOKABLE %1 get%2_%3() const;\n")
                        .arg(type).arg(field->name).arg(elementName);
                propertiesImpl +=
                        QString("%1 %2::get%3_%4() const\n"
                                "{\n"
                                "   QMutexLocker locker(mutex);\n"
                                "   return data.%3[%5];\n"
                                "}\n")
                        .arg(type).arg(info->name).arg(field->name).arg(elementName).arg(elementIndex);
                propertySetters +=
                        QString("    void set%1_%2(%3 value);\n")
                        .arg(field->name).arg(elementName).arg(type);
                propertiesImpl +=
                        QString("void %1::set%2_%3(%4 value)\n"
                                "{\n"
                                "   mutex->lock();\n"
                                "   bool changed = data.%2[%5] != value;\n"
                                "   data.%2[%5] = value;\n"
                                "   mutex->unlock();\n"
                                "   if (changed) emit %2_%3Changed(value);\n"
                                "}\n\n")
                        .arg(info->name).arg(field->name).arg(elementName).arg(type).arg(elementIndex);
                propertyNotifications +=
                        QString("    void %1_%2Changed(%3 value);\n")
                        .arg(field->name).arg(elementName).arg(type);
                propertyNotificationsImpl +=
                        QString("        //if (data.%1[%2] != oldData.%1[%2])\n"
                                "            emit %1_%3Changed(data.%1[%2]);\n")
                        .arg(field->name).arg(elementIndex).arg(elementName);
            }
        } else {
            properties += QString("    Q_PROPERTY(%1 %2 READ get%2 WRITE set%2 NOTIFY %2Changed);\n")
                    .arg(type).arg(field->name);
            propertyGetters +=
                    QString("    Q_INVOKABLE %1 get%2() const;\n")
                    .arg(type).arg(field->name);
            propertiesImpl +=
                    QString("%1 %2::get%3() const\n"
                            "{\n"
                            "   QMutexLocker locker(mutex);\n"
                            "   return data.%3;\n"
                            "}\n")
                    .arg(type).arg(info->name).arg(field->name);
            propertySetters +=
                    QString("    void set%1(%2 value);\n")
                    .arg(field->name).arg(type);
            propertiesImpl +=
                    QString("void %1::set%2(%3 value)\n"
                            "{\n"
                            "   mutex->lock();\n"
                            "   bool changed = data.%2 != value;\n"
                            "   data.%2 = value;\n"
                            "   mutex->unlock();\n"
                            "   if (changed) emit %2Changed(value);\n"
                            "}\n\n")
                    .arg(info->name).arg(field->name).arg(type);
            propertyNotifications +=
                    QString("    void %1Changed(%2 value);\n")
                    .arg(field->name).arg(type);
            propertyNotificationsImpl +=
                    QString("        //if (data.%1 != oldData.%1)\n"
                            "            emit %1Changed(data.%1);\n")
                    .arg(field->name);
        }
    }

    outInclude.replace(QString("$(PROPERTIES)"), properties);
    outInclude.replace(QString("$(PROPERTY_GETTERS)"), propertyGetters);
    outInclude.replace(QString("$(PROPERTY_SETTERS)"), propertySetters);
    outInclude.replace(QString("$(PROPERTY_NOTIFICATIONS)"), propertyNotifications);

    outCode.replace(QString("$(PROPERTIES_IMPL)"), propertiesImpl);
    outCode.replace(QString("$(NOTIFY_PROPERTIES_CHANGED)"), propertyNotificationsImpl);

    // Replace the $(FIELDSINIT) tag
    QString finit;
    for (int n = 0; n < info->fields.length(); ++n)
    {
        // Setup element names
        QString varElemName = info->fields[n]->name + "ElemNames";
        finit.append( QString("    QStringList %1;\n").arg(varElemName) );
        QStringList elemNames = info->fields[n]->elementNames;
        for (int m = 0; m < elemNames.length(); ++m)
            finit.append( QString("    %1.append(\"%2\");\n")
                          .arg(varElemName)
                          .arg(elemNames[m]) );

        // Only for enum types
        if (info->fields[n]->type == FIELDTYPE_ENUM) {
            // Form list of enum names
            finit.append( QString("    QStringList %1EnumOptions = { ")
                          .arg( info->fields[n]->name) );

            QStringList options = info->fields[n]->options;
            for (int m = 0; m < options.length(); ++m)
            {
                finit.append( QString("%1\"%2\"")
                              .arg(QString(m ? ", " : ""))
                              .arg(options[m]) );
            }

            finit.append("};\n");
#if 0
            /* Perhaps use this type in the future, when gcs cleaned up */
            finit.append( QString("    QList<%1Options> %2EnumIndices = { ")
                        .arg( info->fields[n]->name )
                        .arg( info->fields[n]->name ) );
#endif

            finit.append( QString("    QList<int> %2EnumIndices = { ")
                        .arg( info->fields[n]->name ) );

            // Form list of enum values, because they may not be contiguous
            for (int m = 0; m < options.length(); ++m) {
                QString optionName = form_enum_name(QString(),
                        info->fields[n]->name, options[m]);

                QString s = (m != (options.length()-1)) ? "%1, " : "%1";
                finit.append( s.arg(optionName) );
            }

            finit.append("};\n");

            finit.append( QString("    fields.append( new UAVObjectField(QString(\"%1\"), QString(\"%2\"), UAVObjectField::ENUM, %3, %4, %5, QString(\"%6\")));\n")
                          .arg(info->fields[n]->name)
                          .arg(info->fields[n]->units)
                          .arg(varElemName)
                          .arg(info->fields[n]->name + "EnumOptions")
                          .arg(info->fields[n]->name + "EnumIndices")
                          .arg(info->fields[n]->limitValues));
        }
        // For all other types
        else {
            finit.append( QString("    fields.append( new UAVObjectField(QString(\"%1\"), QString(\"%2\"), UAVObjectField::%3, %4, QStringList(), QList<int>(), QString(\"%5\")));\n")
                          .arg(info->fields[n]->name)
                          .arg(info->fields[n]->units)
                          .arg(fieldTypeStrCPPClass[info->fields[n]->type])
                          .arg(varElemName)
                          .arg(info->fields[n]->limitValues));
        }
    }
    outCode.replace(QString("$(FIELDSINIT)"), finit);

    // Replace the $(DATAFIELDINFO) tag
    QString name;
    // To be populated with the enums definition
    QString enums;
    // To be populated with the Q_ENUMS macro
    QString q_enums;
    for (int n = 0; n < info->fields.length(); ++n)
    {
        if(!info->fields[n]->name.isEmpty())
        {
            enums.append(QString("    // Field %1 information\n").arg(info->fields[n]->name));
        }
        // Only for enum types
        if (info->fields[n]->type == FIELDTYPE_ENUM)
        {
            enums.append(QString("    /* Enumeration options for field %1 */\n").arg(info->fields[n]->name));
            enums.append("    typedef enum { ");
            // Go through each option
            QStringList options = info->fields[n]->options;
            for (int m = 0; m < options.length(); ++m) {
                QString optionName = form_enum_name(QString(),
                        info->fields[n]->name, options[m]);
                QString value = QString::number(m);

                if (info->fields[n]->parent) {
                    value = form_enum_name(info->fields[n]->parentObj->name,
                            info->fields[n]->parent->name, options[m]);
                }

                QString s = (m != (options.length()-1)) ? "%1=%2, " : "%1=%2";
                enums.append( s.arg(optionName)
                               .arg(value));

            }
            enums.append( QString(" } %1Options;\n")
                          .arg( info->fields[n]->name ) );
            q_enums.append(QString("    Q_ENUMS(%1Options)\n").arg(info->fields[n]->name));
        }
        // Generate element names (only if field has more than one element)
        if (info->fields[n]->numElements > 1 && !info->fields[n]->defaultElementNames) {
            enums.append(QString("    /* Array element names for field %1 */\n").arg(info->fields[n]->name));
            enums.append("    typedef enum { ");
            // Go through the element names
            QStringList elemNames = info->fields[n]->elementNames;
            for (int m = 0; m < elemNames.length(); ++m) {
                QString s = (m != (elemNames.length()-1)) ? "%1_%2=%3, " : "%1_%2=%3";
                enums.append( s.arg( info->fields[n]->name.toUpper() )
                        .arg( elemNames[m].toUpper() )
                        .arg(m) );
            }
            enums.append( QString(" } %1Elem;\n")
                          .arg( info->fields[n]->name ) );
            q_enums.append(QString("    Q_ENUMS(%1Elem)\n").arg(info->fields[n]->name));
        }
        // Generate array information
        if (info->fields[n]->numElements > 1) {
            enums.append(QString("    /* Number of elements for field %1 */\n").arg(info->fields[n]->name));
            enums.append( QString("    static const quint32 %1_NUMELEM = %2;\n")
                          .arg( info->fields[n]->name.toUpper() )
                          .arg( info->fields[n]->numElements ) );
        }
    }
    outInclude.replace(QString("$(DATAFIELDINFO)"), enums);
    outInclude.replace(QString("$(ENUMS)"),q_enums);
    // Replace the $(INITFIELDS) tag
    QString initfields;

    for (int n = 0; n < info->fields.length(); ++n)
    {
        if (!info->fields[n]->defaultValues.isEmpty() )
        {
            // For non-array fields
            if ( info->fields[n]->numElements == 1)
            {
                if ( info->fields[n]->type == FIELDTYPE_ENUM )
                {
                    initfields.append( QString("    data.%1 = %2;\n")
                                .arg( info->fields[n]->name )
                                .arg( info->fields[n]->options.indexOf( info->fields[n]->defaultValues[0] ) ) );
                }
                else if ( info->fields[n]->type == FIELDTYPE_FLOAT32 )
                {
                    initfields.append( QString("    data.%1 = %2;\n")
                                .arg( info->fields[n]->name )
                                .arg( info->fields[n]->defaultValues[0].toFloat() ) );
                }
                else
                {
                    initfields.append( QString("    data.%1 = %2;\n")
                                .arg( info->fields[n]->name )
                                .arg( info->fields[n]->defaultValues[0].toInt() ) );
                }
            }
            else
            {
                // Initialize all fields in the array
                for (int idx = 0; idx < info->fields[n]->numElements; ++idx)
                {
                    if ( info->fields[n]->type == FIELDTYPE_ENUM ) {
                        initfields.append( QString("    data.%1[%2] = %3;\n")
                                    .arg( info->fields[n]->name )
                                    .arg( idx )
                                    // XXX TODO MPL
                                    .arg( info->fields[n]->options.indexOf( info->fields[n]->defaultValues[idx] ) ) );
                    }
                    else if ( info->fields[n]->type == FIELDTYPE_FLOAT32 ) {
                        initfields.append( QString("    data.%1[%2] = %3;\n")
                                    .arg( info->fields[n]->name )
                                    .arg( idx )
                                    .arg( info->fields[n]->defaultValues[idx].toFloat() ) );
                    }
                    else {
                        initfields.append( QString("    data.%1[%2] = %3;\n")
                                    .arg( info->fields[n]->name )
                                    .arg( idx )
                                    .arg( info->fields[n]->defaultValues[idx].toInt() ) );
                    }
                }
            }
        }
    }

    outCode.replace(QString("$(INITFIELDS)"), initfields);

    // Write the GCS code
    bool res = writeFileIfDiffrent( gcsOutputPath.absolutePath() + "/" + info->namelc + ".cpp", outCode );
    if (!res) {
        cout << "Error: Could not write gcs output files" << endl;
        return false;
    }
    res = writeFileIfDiffrent( gcsOutputPath.absolutePath() + "/" + info->namelc + ".h", outInclude );
    if (!res) {
        cout << "Error: Could not write gcs output files" << endl;
        return false;
    }

    return true;
}
