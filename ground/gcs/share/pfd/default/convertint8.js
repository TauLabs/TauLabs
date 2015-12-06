.pragma library

// workaround for QT bug 37241
function ConvertInt8(a) {
	if (!a.hasOwnProperty("length")) {
		// Doesn't seem like a string.  Let's see if it's a number
		// (e.g. upstream 37241 has been fixed)

		if (!isNaN(a)) {
			return a;
		}

		// Nope.  Don't know what to do here.
		return 0;
	}

	if (a.length < 1) {
		return 0;
	}

	return a.charCodeAt(0);
}
