valgrind \
	--tool=memcheck \
	--leak-check=full \
	--show-reachable=yes \
	--track-fds=yes \
	--track-origins=yes \
	--num-callers=50 \
	--db-attach=no \
	--gen-suppressions=no \
	--suppressions=./ground/gcs/projects/valgrind/memcheck.sup \
	./build/ground/gcs/bin/taulabsgcs.bin
