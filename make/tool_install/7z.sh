# tool_url="http://www.7-zip.org/a/7z920.exe"
#tool_url="http://www.7-zip.org/a/7z920.msi"
tool_url="http://librepilot.github.io/tools/7za.exe"

tool_install_name="bin/7za.exe"
tool_extract_dir="$tools_dir/bin"

bin_dir=$tool_extract_dir

no_extract=true

module_file=bin

#Override
function validate_target { [ "$uname" = Windows ]; }

# Override
function dddownload_and_extract
{
	local full_tool_install_name="$tools_dir/bin/$tool_install_name"
	if ! [ -e  "$full_tool_install_name" ] || $force
	then
		download_and_verify && \
		rm -rf "$full_tool_install_name" && \
		mkdir -p "$(dirname "$full_tool_install_name")" && \
		mv "$downloaded_file" "$full_tool_install_name"
		#msiexec //i "$downloaded_file" //q INSTALLDIR="$tools_dir"
		#cmd //C "$downloaded_file" /S /D="${tools_dir//\//\\}"
	fi
}

