# Dependency-free safetensors loader/writer in C/C++

Secure, dependency-free safetensors loader/writer in portable C/C++.
Code is tested with fuzzer.

## Features

* [x] Load safetensors
  * Load from a file
    * [x] mmap zero-copy load
  * Load from memory
* [x] Save safetensors
  * See [serialize-example.cc](serialize-example.cc) for details.
* [x] BF16 and FP16 support
  * [x] BF16 <-> FLOAT conversion
    * Consider NaN, Inf properly.
  * [x] FP16 <-> FLOAT conversion
    * May not fully consider NaN, Inf properly.
* [x] No C++ thread & exception & RTTI by default.
  * Eliminate issues when writing Language bindings.
  * Better WASM/WASI support
* Portable
  * [x] Windows/VS2022
  * [x] Linux
  * [x] macOS
  * [ ] WASM/WASI, Emscripten
    * Should work. Not tested yet.
  * [ ] Emerging architecture(e.g. RISC-V based AI processor)

## Endianness

Little-endian.

## Requirements

* C++11 and C11 compiler

## Fuzz testing

See [fuzz](fuzz) directory.

## Usage

```cpp
// define only in one *.cc
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

std::string warn, err;
bool ret = safetensors::load_from_file(filename, &st, &warn, &err);

if (warn.size()) {
  std::cout << "WARN: " << warn << "\n";
}

if (!ret) {
  std::cerr << "Failed to load: " << filename << "\n";
  std::cerr << "  ERR: " << err << "\n";

  return false;
}

// Check if data_offsets are valid.
if (!safetensors::validate_data_offsets(st, err)) {
  std::cerr << "Invalid data_offsets\n";
  std::cerr << err << "\n";

  return false;
}

for (size_t i = 0; i < st.tensors.size(); i++) {
  // do something with tensor
}

for (size_t i = 0; i < st.metadata.size(); i++) {
  // do something with __metadata__
}

```

Please see [example.cc](example.cc) for more details.

## Compile

### Windows

```
> vcsetup.bat
```

Then open solution file in `build` folder.

### Linux and macOS

Run makefile

```
$ make
```

or

```
$ ./bootstrap-cmake-linux.sh
$ cd build
$ make
```

## C API

W.I.P.

C API will be provided in `safetensors-c.h` for other language bindings.


## Limitation

* JSON part(header) is up to 100MB.
* ndim is up to 8.

## TODO

* [ ] Strict `shape` size check.
* [ ] Remove `internal::from_chars`(parse number(floating point value) from string)
  * We only need int number parser
* [x] mmap load.
* [x] Save safetensors.
* [ ] Do more tests.
* [ ] validate dict key is valid UTF-8 string.
* [ ] CI build script.
* [ ] C++ STL free?
  * To load safetensor in GPU or Accelerator directly
  * Use nanostl? https://github.com/lighttransport/nanostl

## License

MIT license

### Third-party licenses

* minijson(included in `safetensors.hh`) : MIT license
  * Grisu2(parse floating point value in minijson) : MIT license
  * internal::from_chars : Apache 2.0 license.
* llama.cpp(mmap feature) : MIT license.
* MIOpen(bf16 conversion) : MIT license.
* fp16 conversion: CC0 license. https://gist.github.com/rygorous/2156668
