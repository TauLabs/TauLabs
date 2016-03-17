if [ "$uname" = Linux ]
then
	url_ext="linux.tar.bz2"
elif [ "$uname" = Darwin ]
then
	url_ext="mac.tar.bz2"
elif [ "$uname" = Windows ]
then
	url_ext="win32.zip"
	depends=(7z)
fi

pkgver=4.9_2015_q2_update
pkgdate=20150609
_pkgver=${pkgver//_/-}
_pkgvershort=${_pkgver%-*}
_pkgvershort=${_pkgvershort/-q/q}

tool_url="https://launchpad.net/gcc-arm-embedded/${pkgver%%_*}/${_pkgver}/+download/${tool}-${_pkgvershort/./_}-${pkgdate}-${url_ext}"
tool_md5_url="${tool_url}/+md5"
tool_install_name="${tool}-${_pkgvershort/./_}"
if [ "$uname" = Windows ]
then
	tool_extract_dir=$tools_dir/$tool_install_name
fi

bin_subdir=bin

function validate_target { true; }
