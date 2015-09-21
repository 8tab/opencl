build_dir := build
build_release := build-release
build_debug := build-debug

release:
	mkdir -p $(build_release)
	cmake -B$(build_release) -H. -DCMAKE_BUILD_TYPE=release
	make -C $(build_release)
debug:
	mkdir -p $(build_debug)
	cmake -B$(build_debug) -H. -DCMAKE_BUILD_TYPE=debug
	make -C $(build_debug)
clean:
	rm -rf $(build_debug) $(build_release)
