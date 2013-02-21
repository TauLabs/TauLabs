#
# Regular cron jobs for the taulabs package
#
0 4	* * *	root	[ -x /usr/bin/taulabs_maintenance ] && /usr/bin/taulabs_maintenance
