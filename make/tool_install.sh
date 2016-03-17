#!/bin/bash

# Exit if an error or an unset variable
set -e -u

# make sure unmatched glob gives empty
shopt -s nullglob

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tools_dir=${TOOLS_DIR:-${root_dir}/tools}
downloads_dir=${DL_DIR:-${root_dir}/downloads}
tool_overrides_dir=$root_dir/make/tool_install

batch=${BATCH:-false}
force=false
remove=false
includes=true

for arg in "${@:1}"
do
	[ "$arg" = "-r" ] && remove=true
	[ "$arg" = "-f" ] && force=true
	[ "$arg" = "-n" ] && includes=false

done

tool=${@: -1}

uname=$(uname)
if [[ "$uname" != [LD]* ]]
then
	uname=Windows
fi


# Batch mode
if $batch
then
	CURL_OPTIONS=(--silent -L)
else
	CURL_OPTIONS=(-L)
fi

################################################################################
# Helper functions
################################################################################

function exit_error
{
	error=$?
	echo "${@}"
	exit $error	
}

##  Downloads a file if it doesn't exist
#1  URL
#2  Output filename (optional)
#3+ Additional options to pass to curl
## Sets:
#out_file: path of the downloaded file 
function download_file
{
	out_file="$downloads_dir/${2:-$(basename "$1")}"

	if ! [ -f "$out_file" ]
	then
		mkdir -p "$downloads_dir" && \
		cd "$downloads_dir" && \
		echo "Downloading $1" && \
		curl "${CURL_OPTIONS[@]}" "${@:3}" -o "$out_file" "$1"
	fi
}

## Unzips a file
#1 The file to unzip
#2 The output directory
function zip_extract
{
	unzip "$1" -d "$2"
}

## Extracts a 7zip file
#1 The file to extract
#2 The output directory
function sevenzip_extract
{
	if [ "$uname" = Windows ]
	then
		7za.exe x -o"$2" "$1"
	else
		7zr x -o"$2" "$1"
	fi
}

## Extracts a tar file
#1 The file to extract
#2 The output directory
function tar_extract
{
	tar -xf "$1" -C "$2"
}

##  Extracts a file
#1  File to extract
#2  Extract directory (optional)
no_extract=false # Optional
## Sets:
#out_dir: directory the file was extracted into 
function extract_file
{
	out_dir="${2:-.}"

	echo "Extracting $1"

	mkdir -p "$out_dir" && \
	case "$1" in
		*.zip)
			zip_extract "$1" "$out_dir"
			;;
		*.7z)
			sevenzip_extract "$1" "$out_dir"
			;;
		*.tar*)
			tar_extract "$1" "$out_dir"
			;;
		*)
			if $no_extract
			then
				cp "$1" "$out_dir"
			else
				return 1
			fi
	esac
}

## Verifies an md5 file
#1 md5 file
#2 file to check
function md5_verify_file
{
	if [ "$uname" = Darwin ]
	then
		[[ "$(md5 "$2")" = *"$(awk '{print $1}' "$1")"* ]]
	else
		( cd "$downloads_dir" && md5sum -c "$1" )
	fi
}

################################################################################
# Default functions
################################################################################

function validate_target { false; }

function remove
{
	rm -rf "$tools_dir/$tool_install_name"
	rm -f "$tools_dir/$tool".{sh,mk}
}

declare -a depends=()
function install_deps
{
	# Workaround for set -u and empty array
	for dep in "${depends[@]:+${depends}}"
	do
		BATCH="$batch" "${BASH_SOURCE[0]}" "$dep"
	done && \
	source_includes
}

## Downloads and verifies the tool
## Required:
#tool_url: the url to download the tool from
## Optional:
#tool_md5
#tool_md5_url
function download_and_verify
{
	verified=true
	download_file "$tool_url" && \
	downloaded_file=$out_file && \
	if [ -n "${tool_md5_url:-}" ]
	then
		download_file "$tool_md5_url" "$(basename "$downloaded_file").md5" --silent && \
		if ! md5_verify_file "$out_file" "$downloaded_file"
		then
			mv -f "$downloaded_file"{,.rej} && \
			mv -f "$downloaded_file".md5{,.rej} && \
			verified=false
		fi
	elif [ -n "${tool_md5:-}" ]
	then
		if [[ "$tool_md5"* != "$(cd "$downloads_dir" && md5sum "$downloaded_file")" ]]
		then
			mv -f "$downloaded_file"{,.rej} && \
			verified=false
		fi
	fi && \
	$verified
}

function tool_is_installed { [ -e  "$full_tool_install_name" ] || which "$tool" &>/dev/null; }

## Downloads and extracts the tool
## Required:
#tool_url: the url to download the tool from
#tool_install_name: the directory or file the tool will be installed as
## Optional:
#tool_extract_dir: Directory to extract into (useful if build required)
function download_and_extract
{
	local full_tool_install_name="$tools_dir/$tool_install_name"
	if ! tool_is_installed || $force
	then
		download_and_verify || exit_error "Failed to verify $downloaded_file"
		rm -rf "$full_tool_install_name" && \
		extract_file "$downloaded_file" "${tool_extract_dir:-$tools_dir}"
	fi
}

function build_and_install { true; } # Most tools don't need this step

## Write modules that are included by this script and make
## Optional:
bin_dir=""
bin_subdir="" #
module_file=$tool
function write_modules
{
	if [ -n "$bin_subdir" ]
	then
		bin_dir="$tools_dir/$tool_install_name/$bin_subdir"
	fi

	if [ -n "$bin_dir" ]
	then
		local new_path="$bin_dir"':${PATH}'
		# Write shell module file
		echo 'if [[ ":$PATH:" != *":'"$bin_dir"':"* ]]; then export PATH='"$new_path"'; fi' > "$tools_dir/$module_file".sh
		# Write make module file
		echo 'ifeq ($(findstring :'"$bin_dir"':,:$(PATH):),)' > "$tools_dir/$module_file".mk
		echo "export PATH := $new_path" >> "$tools_dir/$module_file".mk
		echo "endif" >> "$tools_dir/$module_file".mk
	fi
}


function source_includes
{
	if $includes
	then
		for module in "$tools_dir"/*.sh
		do
			source "$module"
		done
	fi
}

################################################################################
# Peform tool install
################################################################################
source_includes || exit_error "failed to source includes"

source "$tool_overrides_dir/${tool}.sh"

if $remove
then
	remove || exit_error "Failed to remove ${tool}"
else
	validate_target || exit_error "${tool} is not a valid target"

	install_deps || exit_error "Failed to install dependencies for ${tool}"

	download_and_extract || exit_error "Failed to download and extract ${tool}"

	build_and_install || exit_error "Failed to build and install ${tool}"

	write_modules || exit_error "Failed to write modules for ${tool}"
fi
