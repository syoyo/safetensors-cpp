rm -rf build
mkdir build

cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -Bbuild -H.
