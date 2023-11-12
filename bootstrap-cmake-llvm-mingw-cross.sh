# cross compile with llvm-mingw
# Assume Ninja is installed on your system
curdir=`pwd`

# Set path to llvm-mingw in env var.
export LLVM_MINGW_DIR=/mnt/data/local/llvm-mingw-20231031-ucrt-ubuntu-20.04-x86_64/

builddir=${curdir}/build-llvm-mingw

rm -rf ${builddir}
mkdir ${builddir}

cd ${builddir} && cmake \
  -DCMAKE_TOOLCHAIN_FILE=${curdir}/cmake/llvm-mingw-cross.cmake \
  -G "Ninja" \
  -DCMAKE_VERBOSE_MAKEFILE=1 \
  ..

cd ${curdir}

