#
# Regular cron jobs for the abovegroundlabs package
#
0 4	* * *	root	[ -x /usr/bin/abovegroundlabs_maintenance ] && /usr/bin/abovegroundlabs_maintenance
