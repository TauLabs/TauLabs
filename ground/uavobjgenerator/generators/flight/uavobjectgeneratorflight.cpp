/**
 ******************************************************************************
 *
 * @file       uavobjectgeneratorflight.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      produce flight code for uavobjects
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

#include "uavobjectgeneratorflight.h"

using namespace std;

bool UAVObjectGeneratorFlight::generate(UAVObjectParser* parser,QString templatepath,QString outputpath) {

    fieldTypeStrC << "int8_t" << "int16_t" << "int32_t" <<"uint8_t"
            <<"uint16_t" << "uint32_t" << "float" << "uint8_t";

    QString flightObjInit,objInc,objFileNames,objNames;
    qint32 sizeCalc;
    flightCodePath = QDir( templatepath + QString("flight/UAVObjects"));
    flightOutputPath = QDir( outputpath + QString("flight") );
    flightOutputPath.mkpath(flightOutputPath.absolutePath());

    flightCodeTemplate = readFile( flightCodePath.absoluteFilePath("uavobjecttemplate.c") );
    flightIncludeTemplate = readFile( flightCodePath.absoluteFilePath("inc/uavobjecttemplate.h") );
    flightInitTemplate = readFile( flightCodePath.absoluteFilePath("uavobjectsinittemplate.c") );
    flightInitIncludeTemplate = readFile( flightCodePath.absoluteFilePath("inc/uavobjectsinittemplate.h") );
    flightVersionTemplate = readFile( flightCodePath.absoluteFilePath("inc/uavoversiontemplate.h") );

    if ( flightCodeTemplate.isNull() || flightIncludeTemplate.isNull() || flightInitTemplate.isNull()) {
            cerr << "Error: Could not open flight template files." << endl;
            return false;
        }

    sizeCalc = 0;
    for (int objidx = 0; objidx < parser->getNumObjects(); ++objidx) {
        ObjectInfo* info=parser->getObjectByIndex(objidx);
        process_object(info);
        flightObjInit.append("#ifdef UAVOBJ_INIT_" + info->namelc +"\r\n");
        flightObjInit.append("    " + info->name + "Initialize();\r\n");
        flightObjInit.append("#endif\r\n");
        objInc.append("#include \"" + info->namelc + ".h\"\r\n");
	objFileNames.append(" " + info->namelc);
	objNames.append(" " + info->name);
	if (parser->getNumBytes(objidx)>sizeCalc) {
		sizeCalc = parser->getNumBytes(objidx);
	}
    }

    // Write the flight object inialization files
    flightInitTemplate.replace( QString("$(OBJINC)"), objInc);
    flightInitTemplate.replace( QString("$(OBJINIT)"), flightObjInit);
    bool res = writeFileIfDiffrent( flightOutputPath.absolutePath() + "/uavobjectsinit.c",
                     flightInitTemplate );
    if (!res) {
        cout << "Error: Could not write flight object init file" << endl;
        return false;
    }

    // Write the flight object initialization header
    flightInitIncludeTemplate.replace( QString("$(SIZECALCULATION)"), QString().setNum(sizeCalc));
    res = writeFileIfDiffrent( flightOutputPath.absolutePath() + "/uavobjectsinit.h",
                     flightInitIncludeTemplate );
    if (!res) {
        cout << "Error: Could not write flight object init header file" << endl;
        return false;
    }

    // Write the flight object initialization header
    flightVersionTemplate.replace( QString("$(UAVOHASH)"), 
		    QString("0x%1").arg(parser->getUavoHash(), 16, 16, QChar('0')));
    res = writeFileIfDiffrent( flightOutputPath.absolutePath() + "/uavoversion.h",
                     flightVersionTemplate );
    if (!res) {
        cout << "Error: Could not write flight object ver header file" << endl;
        return false;
    }

    return true; // if we come here everything should be fine
}

QString UAVObjectGeneratorFlight::form_enum_name(const QString& objName,
        const QString &fieldName, const QString &option) {
    QString s = "%1_%2_%3";

    return s.arg( objName.toUpper() )
            .arg( fieldName.toUpper() )
            .arg( option.toUpper().replace(QRegExp(ENUM_SPECIAL_CHARS), ""));
}

/**
 * Generate the Flight object files
**/
bool UAVObjectGeneratorFlight::process_object(ObjectInfo* info)
{
    if (info == NULL)
        return false;

    // Prepare output strings
    QString outInclude = flightIncludeTemplate;
    QString outCode = flightCodeTemplate;

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
        type = fieldTypeStrC[info->fields[n]->type];
        // Append field
        if ( info->fields[n]->numElements > 1 )
        {
            fields.append( QString("    %1 %2[%3];\r\n").arg(type)
                           .arg(info->fields[n]->name).arg(info->fields[n]->numElements) );
        }
        else
        {
            fields.append( QString("    %1 %2;\r\n").arg(type).arg(info->fields[n]->name) );
        }
    }
    outInclude.replace(QString("$(DATAFIELDS)"), fields);

    // Replace the $(DATAFIELDINFO) tag
    QString enums;
    for (int n = 0; n < info->fields.length(); ++n)
    {
        enums.append(QString("// Field %1 information\r\n").arg(info->fields[n]->name));
        // Only for enum types
        if (info->fields[n]->type == FIELDTYPE_ENUM)
        {
            enums.append(QString("/* Enumeration options for field %1 */\r\n").arg(info->fields[n]->name));
            enums.append("typedef enum { ");
            // Go through each option
            QStringList options = info->fields[n]->options;
            bool const has_parent = info->fields[n]->parent != NULL;
            for (int m = 0; m < options.length(); ++m) {
                QString optionName = form_enum_name(info->name,
                        info->fields[n]->name, options[m]);
                QString value = QString::number(m);

                if (info->fields[n]->parent) {
                    value = form_enum_name(info->fields[n]->parentObj->name,
                            info->fields[n]->parent->name, options[m]);
                }

                // only need to add comma if this is a root options list and this isn't the last option
                QString s = (!has_parent && m == (options.length()-1)) ? "%1=%2" : "%1=%2, ";

                enums.append( s
                                .arg( optionName )
                                .arg( value ) );

            }

            // if this is a child options list, add special enum value to prevent using switch() statements on the generated enum (use root enum instead)
            if (has_parent) enums.append( QString("%1=%2").arg( form_enum_name(info->name, info->fields[n]->name, QString("DONTSWITCHONCHILDENUMS") ) ).arg( 255 ) );

            enums.append( QString(" }  __attribute__((packed)) %1%2Options;\r\n")
                          .arg( info->name )
                          .arg( info->fields[n]->name ) );

            // find topmost parent to get the length of its options field
            FieldInfo_s * topmost_parent = info->fields[n];
            while (topmost_parent->parent) {
                topmost_parent = topmost_parent->parent;
            }

            enums.append(QString("/* Max value of any option in topmost parent %2 of field %1 */\r\n").arg(info->fields[n]->name).arg(topmost_parent->name));
            enums.append( QString("#define %1_%2_GLOBAL_MAXOPTVAL %3\r\n")
                          .arg( info->name.toUpper() )
                          .arg( info->fields[n]->name.toUpper() )
                          .arg( topmost_parent->options.length() - 1 ) );

            // find largest value in this option vector
            int max_optval = 0;
            for (int m = 0; m < info->fields[n]->options.length(); ++m) {
                for (int l = 0; l < topmost_parent->options.length(); ++l) {
                    if (topmost_parent->options[l] == info->fields[n]->options[m])
                        max_optval = max(max_optval, l);
                }
            }

            enums.append(QString("/* Max value of any option in field %1 */\r\n").arg(info->fields[n]->name));
            enums.append( QString("#define %1_%2_MAXOPTVAL %3\r\n")
                          .arg( info->name.toUpper() )
                          .arg( info->fields[n]->name.toUpper() )
                          .arg( max_optval ) );

            /* Validate, for enums only */
            if (info->fields[n]->type == FIELDTYPE_ENUM) {
                enums.append(QString("/* Ensure field %1 contains valid data */\r\n").arg(info->fields[n]->name));
                enums.append(QString("static inline bool %2%3IsValid( %1 Current%3 ) { return Current%3 < %4_%5_MAXOPTVAL; }\r\n")
                            .arg( fieldTypeStrC[info->fields[n]->type] )
                            .arg( info->name )
                            .arg( info->fields[n]->name )
                            .arg( info->name.toUpper() )
                            .arg( info->fields[n]->name.toUpper() ));
            }
        }
        // Generate element names (only if field has more than one element)
        if (info->fields[n]->numElements > 1 && !info->fields[n]->defaultElementNames)
        {
            enums.append(QString("/* Array element names for field %1 */\r\n").arg(info->fields[n]->name));
            enums.append("typedef enum { ");
            // Go through the element names
            QStringList elemNames = info->fields[n]->elementNames;
            for (int m = 0; m < elemNames.length(); ++m)
            {
                QString s = (m != (elemNames.length()-1)) ? "%1_%2_%3=%4, " : "%1_%2_%3=%4";
                enums.append( s
                                .arg( info->name.toUpper() )
                                .arg( info->fields[n]->name.toUpper() )
                                .arg( elemNames[m].toUpper() )
                                .arg(m) );

            }
            enums.append( QString(" } __attribute__((packed)) %1%2Elem;\r\n")
                          .arg( info->name )
                          .arg( info->fields[n]->name ) );
        }
        // Generate array information
        if (info->fields[n]->numElements > 1)
        {
            enums.append(QString("/* Number of elements for field %1 */\r\n").arg(info->fields[n]->name));
            enums.append( QString("#define %1_%2_NUMELEM %3\r\n")
                          .arg( info->name.toUpper() )
                          .arg( info->fields[n]->name.toUpper() )
                          .arg( info->fields[n]->numElements ) );
        }
    }
    outInclude.replace(QString("$(DATAFIELDINFO)"), enums);

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
                    initfields.append( QString("\tdata.%1 = %2;\r\n")
                                .arg( info->fields[n]->name )
                                .arg( info->fields[n]->options.indexOf( info->fields[n]->defaultValues[0] ) ) );
                }
                else if ( info->fields[n]->type == FIELDTYPE_FLOAT32 )
                {
                    initfields.append( QString("\tdata.%1 = %2;\r\n")
                                .arg( info->fields[n]->name )
                                .arg( info->fields[n]->defaultValues[0].toFloat() ) );
                }
                else
                {
                    initfields.append( QString("\tdata.%1 = %2;\r\n")
                                .arg( info->fields[n]->name )
                                .arg( info->fields[n]->defaultValues[0].toInt() ) );
                }
            }
            else
            {
                // Initialize all fields in the array
                for (int idx = 0; idx < info->fields[n]->numElements; ++idx)
                {
                    if ( info->fields[n]->type == FIELDTYPE_ENUM )
                    {
                        initfields.append( QString("\tdata.%1[%2] = %3;\r\n")
                                    .arg( info->fields[n]->name )
                                    .arg( idx )
                                    .arg( info->fields[n]->options.indexOf( info->fields[n]->defaultValues[idx] ) ) );
                    }
                    else if ( info->fields[n]->type == FIELDTYPE_FLOAT32 )
                    {
                        initfields.append( QString("\tdata.%1[%2] = %3;\r\n")
                                    .arg( info->fields[n]->name )
                                    .arg( idx )
                                    .arg( info->fields[n]->defaultValues[idx].toFloat() ) );
                    }
                    else
                    {
                        initfields.append( QString("\tdata.%1[%2] = %3;\r\n")
                                    .arg( info->fields[n]->name )
                                    .arg( idx )
                                    .arg( info->fields[n]->defaultValues[idx].toInt() ) );
                    }
                }
            }
        }
    }
    outCode.replace(QString("$(INITFIELDS)"), initfields);

    // Replace the $(SETGETFIELDS) tag
    QString setgetfields;
    for (int n = 0; n < info->fields.length(); ++n)
    {
        //if (!info->fields[n]->defaultValues.isEmpty() )
        {
            // For non-array fields
            if ( info->fields[n]->numElements == 1)
            {

            	/* Set */
                setgetfields.append( QString("void %2%3Set( %1 *New%3 )\r\n")
							.arg( fieldTypeStrC[info->fields[n]->type] )
							.arg( info->name )
							.arg( info->fields[n]->name ) );
				setgetfields.append( QString("{\r\n") );
				setgetfields.append( QString("\tUAVObjSetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), sizeof(%3));\r\n")
							.arg( info->name )
							.arg( info->fields[n]->name )
							.arg( fieldTypeStrC[info->fields[n]->type] ) );
				setgetfields.append( QString("}\r\n") );

				/* GET */
				setgetfields.append( QString("void %2%3Get( %1 *New%3 )\r\n")
							.arg( fieldTypeStrC[info->fields[n]->type] )
							.arg( info->name )
							.arg( info->fields[n]->name ));
				setgetfields.append( QString("{\r\n") );
				setgetfields.append( QString("\tUAVObjGetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), sizeof(%3));\r\n")
							.arg( info->name )
							.arg( info->fields[n]->name )
							.arg( fieldTypeStrC[info->fields[n]->type] ) );
				setgetfields.append( QString("}\r\n") );

            }
            else
            {

            	/* SET */
				setgetfields.append( QString("void %2%3Set( %1 *New%3 )\r\n")
								.arg( fieldTypeStrC[info->fields[n]->type] )
								.arg( info->name )
								.arg( info->fields[n]->name ) );
				setgetfields.append( QString("{\r\n") );
				setgetfields.append( QString("\tUAVObjSetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), %3*sizeof(%4));\r\n")
								.arg( info->name )
								.arg( info->fields[n]->name )
								.arg( info->fields[n]->numElements )
								.arg( fieldTypeStrC[info->fields[n]->type] ) );
				setgetfields.append( QString("}\r\n") );

				/* GET */
				setgetfields.append( QString("void %2%3Get( %1 *New%3 )\r\n")
								.arg( fieldTypeStrC[info->fields[n]->type] )
								.arg( info->name )
								.arg( info->fields[n]->name ) );
				setgetfields.append( QString("{\r\n") );
				setgetfields.append( QString("\tUAVObjGetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), %3*sizeof(%4));\r\n")
								.arg( info->name )
								.arg( info->fields[n]->name )
								.arg( info->fields[n]->numElements )
								.arg( fieldTypeStrC[info->fields[n]->type] ) );
				setgetfields.append( QString("}\r\n") );
            }
        }
    }
    outCode.replace(QString("$(SETGETFIELDS)"), setgetfields);

    // Replace the $(SETGETFIELDSEXTERN) tag
     QString setgetfieldsextern;
     for (int n = 0; n < info->fields.length(); ++n)
     {
         //if (!info->fields[n]->defaultValues.isEmpty() )
         {

			/* SET */
			setgetfieldsextern.append( QString("extern void %2%3Set( %1 *New%3 );\r\n")
					.arg( fieldTypeStrC[info->fields[n]->type] )
					.arg( info->name )
					.arg( info->fields[n]->name ) );

			/* GET */
			setgetfieldsextern.append( QString("extern void %2%3Get( %1 *New%3 );\r\n")
					.arg( fieldTypeStrC[info->fields[n]->type] )
					.arg( info->name )
					.arg( info->fields[n]->name ) );
         }
     }
     outInclude.replace(QString("$(SETGETFIELDSEXTERN)"), setgetfieldsextern);

    // Write the flight code
    bool res = writeFileIfDiffrent( flightOutputPath.absolutePath() + "/" + info->namelc + ".c", outCode );
    if (!res) {
        cout << "Error: Could not write flight code files" << endl;
        return false;
    }

    res = writeFileIfDiffrent( flightOutputPath.absolutePath() + "/" + info->namelc + ".h", outInclude );
    if (!res) {
        cout << "Error: Could not write flight include files" << endl;
        return false;
    }

    return true;
}


