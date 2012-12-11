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

/**
 * @brief Generate UAVOs for GCS code
 * @param XML parser that contains parsed data
 * @param templatepath path for code templates
 * @param outputpath output path
 * @return true if everything went fine
 */
bool UAVObjectGeneratorGCS::generate(UAVObjectParser* parser,QString templatepath,QString outputpath) {

    fieldTypeStrCPP << "struct" << "qint8" << "qint16" << "qint32" <<
        "quint8" << "quint16" << "quint32" << "float" << "quint8";

    fieldTypeStrCPPClass <<"STRUCT"<< "INT8" << "INT16" << "INT32"
        << "UINT8" << "UINT16" << "UINT32" << "FLOAT32" << "ENUM";

    gcsCodePath = QDir( templatepath + QString(GCS_CODE_DIR));
    gcsOutputPath = QDir( outputpath + QString("gcs") );
    gcsOutputPath.mkpath(gcsOutputPath.absolutePath());

    gcsCodeTemplate = readFile( gcsCodePath.absoluteFilePath("uavobjecttemplate.cpp") );
    gcsIncludeTemplate = readFile( gcsCodePath.absoluteFilePath("uavobjecttemplate.h") );
    QString gcsInitTemplate = readFile( gcsCodePath.absoluteFilePath("uavobjectsinittemplate.cpp") );

    if (gcsCodeTemplate.isEmpty() || gcsIncludeTemplate.isEmpty() || gcsInitTemplate.isEmpty()) {
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

    return true; // if we come here everything should be fine
}

/**
 * @brief Generate the GCS code for one UAVO
 * @param info the object to process
 * @return true if everything went fine
 */
bool UAVObjectGeneratorGCS::process_object(ObjectInfo* info)
{
    if (info == NULL)
        return false;

    bool res=true;

    // Prepare output strings
    QString outInclude = gcsIncludeTemplate;
    QString outCode = gcsCodeTemplate;

    // Replace common tags
    res = res && replaceCommonTags(outInclude, info);
    res = res && replaceCommonTags(outCode, info);

    // Replace the $(DATAFIELDS) tag
    QString fields = QString("");
    res = res && generateStructDefinitions(info->field,fields);
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

    for (int n = 0; n < info->field->childrenFields.length(); ++n)
    {
        FieldInfo *childField = info->field->childrenFields[n];

        if (reservedProperties.contains(childField->name))
            continue;

        // Determine type
        QString type = fieldType(childField);
        // Append field
        if ( childField->numElements > 1 ) {
            //add both field(elementIndex)/setField(elemntIndex,value) and field_element properties
            //field_element is more convenient if only certain element is used
            //and much easier to use from the qml side
            propertyGetters +=
                    QString("    Q_INVOKABLE %1 get%2(quint32 index) const;\n")
                    .arg(type).arg(childField->name);
            propertiesImpl +=
                    QString("%1 %2::get%3(quint32 index) const\n"
                            "{\n"
                            "   QMutexLocker locker(mutex);\n"
                            "   return data.%3[index];\n"
                            "}\n")
                    .arg(type).arg(info->name).arg(childField->name);
            propertySetters +=
                    QString("    void set%1(quint32 index, %2 value);\n")
                    .arg(childField->name).arg(type);
            propertiesImpl +=
                    QString("void %1::set%2(quint32 index, %3 value)\n"
                            "{\n"
                            "   mutex->lock();\n"
                            "   bool changed = data.%2[index] != value;\n"
                            "   data.%2[index] = value;\n"
                            "   mutex->unlock();\n"
                            "   if (changed) emit %2Changed(index,value);\n"
                            "}\n\n")
                    .arg(info->name).arg(childField->name).arg(type);
            propertyNotifications +=
                    QString("    void %1Changed(quint32 index, %2 value);\n")
                    .arg(childField->name).arg(type);

            for (int elementIndex = 0; elementIndex < childField->numElements; elementIndex++) {
                QString elementName = childField->elementNames[elementIndex];
                properties += QString("    Q_PROPERTY(%1 %2 READ get%2 WRITE set%2 NOTIFY %2Changed);\n")
                        .arg(type).arg(childField->name+"_"+elementName);
                propertyGetters +=
                        QString("    Q_INVOKABLE %1 get%2_%3() const;\n")
                        .arg(type).arg(childField->name).arg(elementName);
                propertiesImpl +=
                        QString("%1 %2::get%3_%4() const\n"
                                "{\n"
                                "   QMutexLocker locker(mutex);\n"
                                "   return data.%3[%5];\n"
                                "}\n")
                        .arg(type).arg(info->name).arg(childField->name).arg(elementName).arg(elementIndex);
                propertySetters +=
                        QString("    void set%1_%2(%3 value);\n")
                        .arg(childField->name).arg(elementName).arg(type);
                propertiesImpl +=
                        QString("void %1::set%2_%3(%4 value)\n"
                                "{\n"
                                "   mutex->lock();\n"
                                "   bool changed = data.%2[%5] != value;\n"
                                "   data.%2[%5] = value;\n"
                                "   mutex->unlock();\n"
                                "   if (changed) emit %2_%3Changed(value);\n"
                                "}\n\n")
                        .arg(info->name).arg(childField->name).arg(elementName).arg(type).arg(elementIndex);
                propertyNotifications +=
                        QString("    void %1_%2Changed(%3 value);\n")
                        .arg(childField->name).arg(elementName).arg(type);
                propertyNotificationsImpl +=
                        QString("        //if (data.%1[%2] != oldData.%1[%2])\n"
                                "            emit %1_%3Changed(data.%1[%2]);\n")
                        .arg(childField->name).arg(elementIndex).arg(elementName);
            }
        } else {
            properties += QString("    Q_PROPERTY(%1 %2 READ get%2 WRITE set%2 NOTIFY %2Changed);\n")
                    .arg(type).arg(childField->name);
            propertyGetters +=
                    QString("    Q_INVOKABLE %1 get%2() const;\n")
                    .arg(type).arg(childField->name);
            propertiesImpl +=
                    QString("%1 %2::get%3() const\n"
                            "{\n"
                            "   QMutexLocker locker(mutex);\n"
                            "   return data.%3;\n"
                            "}\n")
                    .arg(type).arg(info->name).arg(childField->name);
            propertySetters +=
                    QString("    void set%1(%2 value);\n")
                    .arg(childField->name).arg(type);
            propertiesImpl +=
                    QString("void %1::set%2(%3 value)\n"
                            "{\n"
                            "   mutex->lock();\n"
                            "   bool changed = data.%2 != value;\n"
                            "   data.%2 = value;\n"
                            "   mutex->unlock();\n"
                            "   if (changed) emit %2Changed(value);\n"
                            "}\n\n")
                    .arg(info->name).arg(childField->name).arg(type);
            propertyNotifications +=
                    QString("    void %1Changed(%2 value);\n")
                    .arg(childField->name).arg(type);
            propertyNotificationsImpl +=
                    QString("        //if (data.%1 != oldData.%1)\n"
                            "            emit %1Changed(data.%1);\n")
                    .arg(childField->name);
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
    for (int n = 0; n < info->field->childrenFields.length(); ++n)
    {
        // Setup element names
        QString varElemName = info->field->childrenFields[n]->name + "ElemNames";
        finit.append( QString("    QStringList %1;\n").arg(varElemName) );
        QStringList elemNames = info->field->childrenFields[n]->elementNames;
        for (int m = 0; m < elemNames.length(); ++m)
            finit.append( QString("    %1.append(\"%2\");\n")
                          .arg(varElemName)
                          .arg(elemNames[m]) );

        // Only for enum types
        if (info->field->childrenFields[n]->type == FIELDTYPE_ENUM) {
            QString varOptionName = info->field->childrenFields[n]->name + "EnumOptions";
            finit.append( QString("    QStringList %1;\n").arg(varOptionName) );
            QStringList options = info->field->childrenFields[n]->options;
            for (int m = 0; m < options.length(); ++m)
            {
                finit.append( QString("    %1.append(\"%2\");\n")
                              .arg(varOptionName)
                              .arg(options[m]) );
            }
            finit.append( QString("    fields.append( new UAVObjectField(QString(\"%1\"), QString(\"%2\"), UAVObjectField::ENUM, %3, %4, QString(\"%5\")));\n")
                          .arg(info->field->childrenFields[n]->name)
                          .arg(info->field->childrenFields[n]->units)
                          .arg(varElemName)
                          .arg(varOptionName)
                          .arg(info->field->childrenFields[n]->limitValues));
        }
        // For all other types
        else {
            finit.append( QString("    fields.append( new UAVObjectField(QString(\"%1\"), QString(\"%2\"), UAVObjectField::%3, %4, QStringList(), QString(\"%5\")));\n")
                          .arg(info->field->childrenFields[n]->name)
                          .arg(info->field->childrenFields[n]->units)
                          .arg(fieldTypeStrCPPClass[info->field->childrenFields[n]->type])
                          .arg(varElemName)
                          .arg(info->field->childrenFields[n]->limitValues));
        }
    }
    outCode.replace(QString("$(FIELDSINIT)"), finit);

    // Replace the $(DATAFIELDINFO) tag
    QString enums = QString("");
    res = res && generateEnumDefinitions(info->field,enums);
    outInclude.replace(QString("$(DATAFIELDINFO)"), enums);

    // Replace the $(INITFIELDS) tag
    QString initfields;
    for (int n = 0; n < info->field->childrenFields.length(); ++n)
    {
        if (!info->field->childrenFields[n]->defaultValues.isEmpty() )
        {
            // For non-array fields
            if ( info->field->childrenFields[n]->numElements == 1)
            {
                if ( info->field->childrenFields[n]->type == FIELDTYPE_ENUM )
                {
                    initfields.append( QString("    data.%1 = %2;\n")
                                .arg( info->field->childrenFields[n]->name )
                                .arg( info->field->childrenFields[n]->options.indexOf( info->field->childrenFields[n]->defaultValues[0] ) ) );
                }
                else if ( info->field->childrenFields[n]->type == FIELDTYPE_FLOAT32 )
                {
                    initfields.append( QString("    data.%1 = %2;\n")
                                .arg( info->field->childrenFields[n]->name )
                                .arg( info->field->childrenFields[n]->defaultValues[0].toFloat() ) );
                }
                else
                {
                    initfields.append( QString("    data.%1 = %2;\n")
                                .arg( info->field->childrenFields[n]->name )
                                .arg( info->field->childrenFields[n]->defaultValues[0].toInt() ) );
                }
            }
            else
            {
                // Initialize all fields in the array
                for (int idx = 0; idx < info->field->childrenFields[n]->numElements; ++idx)
                {
                    if ( info->field->childrenFields[n]->type == FIELDTYPE_ENUM ) {
                        initfields.append( QString("    data.%1[%2] = %3;\n")
                                    .arg( info->field->childrenFields[n]->name )
                                    .arg( idx )
                                    .arg( info->field->childrenFields[n]->options.indexOf( info->field->childrenFields[n]->defaultValues[idx] ) ) );
                    }
                    else if ( info->field->childrenFields[n]->type == FIELDTYPE_FLOAT32 ) {
                        initfields.append( QString("    data.%1[%2] = %3;\n")
                                    .arg( info->field->childrenFields[n]->name )
                                    .arg( idx )
                                    .arg( info->field->childrenFields[n]->defaultValues[idx].toFloat() ) );
                    }
                    else {
                        initfields.append( QString("    data.%1[%2] = %3;\n")
                                    .arg( info->field->childrenFields[n]->name )
                                    .arg( idx )
                                    .arg( info->field->childrenFields[n]->defaultValues[idx].toInt() ) );
                    }
                }
            }
        }
    }

    outCode.replace(QString("$(INITFIELDS)"), initfields);

    if (!res) {
        cout << "Error: Could not substitute tags" << endl;
        return false;
    }

    // Write the GCS code
    res = res &&  writeFileIfDiffrent( gcsOutputPath.absolutePath() + "/" + info->namelc + ".cpp", outCode );
    if (!res) {
        cout << "Error: Could not write gcs output files" << endl;
        return false;
    }
    res = res &&  writeFileIfDiffrent( gcsOutputPath.absolutePath() + "/" + info->namelc + ".h", outInclude );
    if (!res) {
        cout << "Error: Could not write gcs output files" << endl;
        return false;
    }

    return res;
}


/**
 * @brief Retrieves the path to a specific field in the field arborescence
 * @param field
 * @return a list representing the path to the field. The root is the first element of the list, the field is the last element
 */
QStringList UAVObjectGeneratorGCS::fieldPath(FieldInfo* field) {
    QStringList result = QStringList();
    FieldInfo* current=field;
    while(current!=NULL) {
        result.prepend(current->name);
        current = current->parentField;
    }
    return result;
}

/**
 * @brief Generate the field type
 * @param field
 * @return the C type for the field, even if it is a struct field
 */
QString UAVObjectGeneratorGCS::fieldType(FieldInfo* field) {

    return (field->type == FIELDTYPE_STRUCT)?
                fieldPath(field).join("").replace(QRegExp(ENUM_SPECIAL_CHARS), "").append("Data"):
                fieldTypeStrCPP[field->type] ;
}

/**
 * @brief Generate the field descriptor
 * @param field
 * @return a string like "int8 blabla[10]"
 */
QString UAVObjectGeneratorGCS::fieldDescriptor(FieldInfo* field) {
    return (field->numElements > 1)?
                QString("%1 %2[%3]").arg(fieldType(field)).arg(field->name).arg(field->numElements):
                QString("%1 %2").arg(fieldType(field)).arg(field->name);
}



/**
 * @brief Generate the structs definitions string, recursively
 * @param field field to process
 * @param datafields string to prepend with the definition of the field
 * @return true if everything went fine
 */
bool UAVObjectGeneratorGCS::generateStructDefinitions(FieldInfo* field, QString& datafields)
{

    bool res = true;

    QString buffer = QString("");
    //for each subfield add its declaration in the struct declaration for the field
    foreach(FieldInfo* childField, field->childrenFields) {
            buffer.append( QString("    %1;\r\n").arg(fieldDescriptor(childField) ));
    }

    QString cStructName = fieldPath(field).join("").replace(QRegExp(ENUM_SPECIAL_CHARS), "") + QString("Data");
    datafields.prepend(QString("typedef struct {\r\n") + buffer +  QString("} __attribute__((packed)) ") + cStructName +QString(";\r\n\r\n"));


    // for each subfield of struct type, prepend datafields with the corresponding struct type definition
    foreach(FieldInfo* childField, field->childrenFields) {
        if(childField->type == FIELDTYPE_STRUCT) {
            res = res && generateStructDefinitions(childField, datafields);
        }
    }

    return res;
}

/**
 * @brief Generate the structs definitions string, recursively
 * @param field field to process
 * @param datafields string to prepend with the definition of the field
 * @return true if everything went fine
 */
bool UAVObjectGeneratorGCS::generateEnumDefinitions(FieldInfo* field, QString& enums)
{

    bool res = true;

    QString buffer = QString("");
    foreach(FieldInfo* childField, field->childrenFields) {

        // For enums
        if (childField->type == FIELDTYPE_ENUM)
        {

            buffer.append(QString("// Enumeration options for field %1 \r\n").arg(fieldPath(childField).join(QString("."))));
            buffer.append("typedef enum { ");
            // Go through each option
            QStringList options = QStringList();
            int i = 0;
            foreach(QString option,childField->options) {
                options.append(QString("%1_%2=%3").arg(fieldPath(childField).join(QString("_")).toUpper()).arg(option.toUpper().replace(QRegExp(ENUM_SPECIAL_CHARS), "")).arg(i));
                i++;
            }
            buffer.append(options.join(QString(", ")));
            buffer.append( QString(" } %1Options;\r\n").arg(fieldPath(childField).join(QString(""))));
        }


        // Generate element names (only if field has more than one element)
        if (childField->numElements > 1 && !childField->defaultElementNames)
        {

            buffer.append(QString("// Array element names for field %1 \r\n").arg(fieldPath(childField).join(QString("."))));
            buffer.append("typedef enum { ");
            // Go through the element names
            QStringList elementNames = QStringList();
            int i = 0;
            foreach(QString elementName,childField->elementNames) {
                elementNames.append(QString("%1_%2=%3").arg(fieldPath(childField).join(QString("_")).toUpper()).arg(elementName.toUpper().replace(QRegExp(ENUM_SPECIAL_CHARS), "")).arg(i));
                i++;
            }
            buffer.append(elementNames.join(QString(", ")));
            buffer.append( QString(" } %1Elem;\r\n").arg(fieldPath(childField).join(QString(""))));
        }
        // Generate array information
        if (childField->numElements > 1)
        {
            buffer.append(QString("// Number of elements for field %1 \r\n").arg(fieldPath(childField).join(QString("."))));
            buffer.append( QString("    static const quint32 %1_NUMELEM = %2;\n")
                           .arg(fieldPath(childField).join(QString("_")).toUpper() )
                           .arg( childField->numElements ));
        }
    }
    enums.prepend(buffer);


    // process subfields
    foreach(FieldInfo* childField, field->childrenFields) {
        if(childField->type == FIELDTYPE_STRUCT)
            res = res && generateEnumDefinitions(childField, enums);
    }

    return res;
}
