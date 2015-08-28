.pragma library

// workaround for QT bug 37241
function ConvertInt8(a) {
	if (a.length < 1) { return 0; }
	return a.charCodeAt(0);
}
