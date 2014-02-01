/**
 ******************************************************************************
 * @addtogroup TauLabsModules TauLabs Modules
 * @{ 
 * @addtogroup PicoC Interpreter Module
 * @{ 
 *
 * @file       picoc_clibrary.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      c-interpreter module for autonomous user programmed tasks
 *             replacement for original clibrary.c
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


// conditional compilation of the module
#include "pios.h"
#ifdef PIOS_INCLUDE_PICOC

#include "openpilot.h"
#include "picoc_port.h"

/* picoc mini standard C library - provides an optional tiny C standard library 
 * if BUILTIN_MINI_STDLIB is defined */

#include "picoc.h"
#include "interpreter.h"

/* global initialisation for libraries */
void LibraryInit(Picoc *pc)
{
	/* define the version number macro */
	pc->VersionString = TableStrRegister(pc, PICOC_VERSION);
	VariableDefinePlatformVar(pc, NULL, "PICOC_VERSION", pc->CharPtrType, (union AnyValue *)&pc->VersionString, FALSE);
}

/* add a library */
void LibraryAdd(Picoc *pc, struct Table *GlobalTable, const char *LibraryName, struct LibraryFunction *FuncList)
{
	struct ParseState Parser;
	int Count;
	char *Identifier;
	struct ValueType *ReturnType;
	struct Value *NewValue;
	void *Tokens;
	char *IntrinsicName = TableStrRegister(pc, "c library");

	/* read all the library definitions */
	for (Count = 0; FuncList[Count].Prototype != NULL; Count++)
	{
		Tokens = LexAnalyse(pc, IntrinsicName, FuncList[Count].Prototype, strlen((char *)FuncList[Count].Prototype), NULL);
		LexInitParser(&Parser, pc, FuncList[Count].Prototype, Tokens, IntrinsicName, TRUE, FALSE);
		TypeParse(&Parser, &ReturnType, &Identifier, NULL);
		NewValue = ParseFunctionDefinition(&Parser, ReturnType, Identifier);
		NewValue->Val->FuncDef.Intrinsic = FuncList[Count].Func;
		HeapFreeMem(pc, Tokens);
	}
}

/* print a type to a stream without using printf/sprintf */
void PrintType(struct ValueType *Typ, IOFILE *Stream)
{
	switch (Typ->Base)
	{
		case TypeVoid:			PrintStr("void", Stream); break;
		case TypeInt:			PrintStr("int", Stream); break;
		case TypeShort:			PrintStr("short", Stream); break;
		case TypeChar:			PrintStr("char", Stream); break;
		case TypeLong:			PrintStr("long", Stream); break;
		case TypeUnsignedInt:	PrintStr("unsigned int", Stream); break;
		case TypeUnsignedShort:	PrintStr("unsigned short", Stream); break;
		case TypeUnsignedLong:	PrintStr("unsigned long", Stream); break;
		case TypeUnsignedChar:	PrintStr("unsigned char", Stream); break;
#ifndef NO_FP
		case TypeFP:			PrintStr("double", Stream); break;
#endif
		case TypeFunction:		PrintStr("function", Stream); break;
		case TypeMacro:			PrintStr("macro", Stream); break;
		case TypePointer:		if (Typ->FromType) PrintType(Typ->FromType, Stream); PrintCh('*', Stream); break;
		case TypeArray:			PrintType(Typ->FromType, Stream); PrintCh('[', Stream); if (Typ->ArraySize != 0) PrintSimpleInt(Typ->ArraySize, Stream); PrintCh(']', Stream); break;
		case TypeStruct:		PrintStr("struct ", Stream); PrintStr( Typ->Identifier, Stream); break;
		case TypeUnion:			PrintStr("union ", Stream); PrintStr(Typ->Identifier, Stream); break;
		case TypeEnum:			PrintStr("enum ", Stream); PrintStr(Typ->Identifier, Stream); break;
		case TypeGotoLabel:		PrintStr("goto label ", Stream); break;
		case Type_Type:			PrintStr("type ", Stream); break;
	}
}


#ifdef BUILTIN_MINI_STDLIB

/* 
 * This is a simplified standard library for small embedded systems. It doesn't require
 * a system stdio library to operate.
 *
 * A more complete standard library for larger computers is in the library_XXX.c files.
 */

static int TRUEValue = 1;
static int ZeroValue = 0;

void BasicIOInit(Picoc *pc)
{
	pc->CStdOutBase.Putch = &PlatformPutc;
	pc->CStdOut = &pc->CStdOutBase;
}

/* initialise the C library */
void CLibraryInit(Picoc *pc)
{
	/* define some constants */
	VariableDefinePlatformVar(pc, NULL, "NULL", &pc->IntType, (union AnyValue *)&ZeroValue, FALSE);
	VariableDefinePlatformVar(pc, NULL, "TRUE", &pc->IntType, (union AnyValue *)&TRUEValue, FALSE);
	VariableDefinePlatformVar(pc, NULL, "FALSE", &pc->IntType, (union AnyValue *)&ZeroValue, FALSE);
}

/* stream for writing into strings */
void SPutc(unsigned char Ch, union OutputStreamInfo *Stream)
{
	struct StringOutputStream *Out = &Stream->Str;
	*Out->WritePos++ = Ch;
}

/* print a character to a stream without using printf/sprintf */
void PrintCh(char OutCh, struct OutputStream *Stream)
{
	(*Stream->Putch)(OutCh, &Stream->i);
}

/* print a string to a stream without using printf/sprintf */
void PrintStr(const char *Str, struct OutputStream *Stream)
{
	while (*Str != 0)
		PrintCh(*Str++, Stream);
}

/* print an unsigned integer to a stream without using printf/sprintf */
void PrintUnsigned(unsigned long Num, unsigned int Base, int FieldWidth, int ZeroPad, int LeftJustify, struct OutputStream *Stream)
{
	char Result[33];
	char Format[8];
	char *FPos = Format;

	/* build format string */
	*FPos++ = '%';
	if (LeftJustify)
		*FPos++ = '-';
	if (ZeroPad)
		*FPos++ = '0';
	if ((FieldWidth > 0) && (FieldWidth < 20))
		FPos += snprintf(FPos, 2, "%d", FieldWidth);
	switch (Base)
	{
	case 16:
		*FPos++ = 'x';
		break;
	default:
		*FPos++ = 'd';
	}
	*FPos++ = '\0';

	snprintf(Result, sizeof(Result), Format, Num);
	PrintStr(Result, Stream);
}

/* print an integer to a stream without using printf/sprintf */
void PrintSimpleInt(long Num, struct OutputStream *Stream)
{
	PrintInt(Num, -1, FALSE, FALSE, Stream);
}

/* print an integer to a stream without using printf/sprintf */
void PrintInt(long Num, int FieldWidth, int ZeroPad, int LeftJustify, struct OutputStream *Stream)
{
	if (Num < 0)
	{
		PrintCh('-', Stream);
		Num = -Num;
	if (FieldWidth != 0)
		FieldWidth--;
	}
	PrintUnsigned((unsigned long)Num, 10, FieldWidth, ZeroPad, LeftJustify, Stream);
}

#ifndef NO_FP
/* print a double to a stream without using printf/sprintf */
void PrintFP(double Num, struct OutputStream *Stream)
{
	int Exponent = 0;
	int MaxDecimal;

	if (Num < 0)
	{
		PrintCh('-', Stream);
		Num = -Num;
	}

	if (Num >= 1e7)
		Exponent = log10(Num);
	else if (Num <= 1e-7 && Num != 0.0)
		Exponent = log10(Num) - 0.999999999;

	Num /= pow(10.0, Exponent);
	PrintInt((long)Num, 0, FALSE, FALSE, Stream);
	PrintCh('.', Stream);
	Num = (Num - (long)Num) * 10;
	if (abs(Num) >= 1e-7)
	{
		for (MaxDecimal = 6; MaxDecimal > 0 && abs(Num) >= 1e-7; Num = (Num - (long)(Num + 1e-7)) * 10, MaxDecimal--)
			PrintCh('0' + (long)(Num + 1e-7), Stream);
	}
	else
		PrintCh('0', Stream);

	if (Exponent != 0)
	{
		PrintCh('e', Stream);
		PrintInt(Exponent, 0, FALSE, FALSE, Stream);
	}
}
#endif

/* intrinsic functions made available to the language */
void GenericPrintf(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs, struct OutputStream *Stream)
{
	char *FPos;
	struct Value *NextArg = Param[0];
	struct ValueType *FormatType;
	int ArgCount = 1;
	int LeftJustify = FALSE;
	int ZeroPad = FALSE;
	int FieldWidth = 0;
	char *Format = Param[0]->Val->Pointer;
	Picoc *pc = Parser->pc;

	for (FPos = Format; *FPos != '\0'; FPos++)
	{
		if (*FPos == '%')
		{
			FPos++;
			FieldWidth = 0;
			if (*FPos == '-')
			{	/* a leading '-' means left justify */
				LeftJustify = TRUE;
				FPos++;
			}

			if (*FPos == '0')
			{	/* a leading zero means zero pad a decimal number */
				ZeroPad = TRUE;
				FPos++;
			}

			/* get any field width in the format */
			while (isdigit((int)*FPos))
				FieldWidth = FieldWidth * 10 + (*FPos++ - '0');

			/* now check the format type */
			switch (*FPos)
			{
			case 's':
				FormatType = pc->CharPtrType;
				break;
			case 'd':
			case 'u':
			case 'x':
			case 'c':
				FormatType = &pc->IntType;
				break;
#ifndef NO_FP
			case 'f':
				FormatType = &pc->FPType;
				break;
#endif
			case '%':
				PrintCh('%', Stream);
				FormatType = NULL;
				break;
			case '\0':
				FPos--;
				FormatType = NULL;
				break;
			default:
				PrintCh(*FPos, Stream);
				FormatType = NULL;
			}

			if (FormatType != NULL)
			{	/* we have to format something */
				if (ArgCount >= NumArgs)
				{	/* not enough parameters for format */
					PrintStr("XXX", Stream);
				}
				else
				{
					NextArg = (struct Value *)((char *)NextArg + MEM_ALIGN(sizeof(struct Value) + TypeStackSizeValue(NextArg)));
					if (NextArg->Typ != FormatType && 
						!((FormatType == &pc->IntType || *FPos == 'f') && IS_NUMERIC_COERCIBLE(NextArg)) &&
						!(FormatType == pc->CharPtrType && (NextArg->Typ->Base == TypePointer || (NextArg->Typ->Base == TypeArray && NextArg->Typ->FromType->Base == TypeChar) ) ) )
					{	/* bad type for format */
						PrintStr("XXX", Stream);
					}
					else
					{
						switch (*FPos)
						{
						case 's':
							{
								char *Str;
								if (NextArg->Typ->Base == TypePointer)
									Str = NextArg->Val->Pointer;
								else
									Str = &NextArg->Val->ArrayMem[0];
								if (Str == NULL)
									PrintStr("NULL", Stream);
								else
									PrintStr(Str, Stream); 
							}
							break;
						case 'd':
							PrintInt(ExpressionCoerceInteger(NextArg), FieldWidth, ZeroPad, LeftJustify, Stream);
							break;
						case 'u':
							PrintUnsigned(ExpressionCoerceUnsignedInteger(NextArg), 10, FieldWidth, ZeroPad, LeftJustify, Stream);
							break;
						case 'x':
							PrintUnsigned(ExpressionCoerceUnsignedInteger(NextArg), 16, FieldWidth, ZeroPad, LeftJustify, Stream);
							break;
						case 'c':
							PrintCh(ExpressionCoerceUnsignedInteger(NextArg), Stream);
							break;
#ifndef NO_FP
						case 'f':
							PrintFP(ExpressionCoerceFP(NextArg), Stream);
							break;
#endif
						}
					}
				}
				ArgCount++;
			}
		}
		else
			PrintCh(*FPos, Stream);
	}
}

/* printf(): print to console output */
void LibPrintf(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	struct OutputStream ConsoleStream;

	ConsoleStream.Putch = &PlatformPutc;
	GenericPrintf(Parser, ReturnValue, Param, NumArgs, &ConsoleStream);
}

/* sprintf(): print to a string */
void LibSPrintf(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	struct OutputStream StrStream;

	StrStream.Putch = &SPutc;
	StrStream.i.Str.Parser = Parser;
	StrStream.i.Str.WritePos = Param[0]->Val->Pointer;

	GenericPrintf(Parser, ReturnValue, Param+1, NumArgs-1, &StrStream);
	PrintCh(0, &StrStream);
	ReturnValue->Val->Pointer = *Param;
}

/* get a line of input. protected from buffer overrun */
void LibGets(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->Pointer = PlatformGetLine(Param[0]->Val->Pointer, GETS_BUF_MAX, NULL);
	if (ReturnValue->Val->Pointer != NULL)
	{
		char *EOLPos = strchr(Param[0]->Val->Pointer, '\n');
		if (EOLPos != NULL)
			*EOLPos = '\0';
	}
}

void LibGetc(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	ReturnValue->Val->Integer = PlatformGetCharacter();
}

void LibPutc(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	PlatformPutc(Param[0]->Val->Integer, NULL);
	ReturnValue->Val->Integer = Param[0]->Val->Integer;
}

void LibExit(struct ParseState *Parser, struct Value *ReturnValue, struct Value **Param, int NumArgs)
{
	PlatformExit(Parser->pc, Param[0]->Val->Integer);
}

/* list of all library functions and their prototypes */
struct LibraryFunction CLibrary[] =
{
	{ LibPrintf,	"void printf(char *, ...);" },
	{ LibSPrintf,	"char *sprintf(char *, char *, ...);" },
	{ LibGets,		"char *gets(char *);" },
	{ LibGetc,		"int getchar();" },
	{ LibPutc,		"int putchar(int);" },
	{ LibExit,		"void exit(int);" },
	{ NULL, NULL }
};

#endif /* BUILTIN_MINI_STDLIB */

#endif /* PIOS_INCLUDE_PICOC */

/**
 * @}
 * @}
 */
