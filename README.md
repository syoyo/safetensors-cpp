# Header-only safetensors loader in C/C++

Secure, header-only safetensors loader and writer in portable C/C++.
Code is tested with fuzzer.

## Requirements

* C++11 and C11 compiler

## Fuzz testing

See [fuzz](fuzz) directory.

## Usage

```
// define only in one *.cc
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

T.B.W.
```

## C API

W.I.P.

C API will be provided in `safetensors-c.h` for other language bindings.


## License

MIT license

### Third-party licenses

* minijson(included in `safetensors.hh`) : MIT license
* Grisu2 : MIT license
