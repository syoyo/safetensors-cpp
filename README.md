# Header-only safetensors loader in C++

Secure, header-only safetensors loader and writer in portable C++.
Code is tested with fuzzer.

## Requirements

* C++11 compiler

## Fuzz testing

See [fuzz](fuzz) directory.

## Usage

```
// define only in one *.cc
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

T.B.W.
```


## License

MIT license

### Third-party licenses

* minijson(included in `safetensors.hh`) : MIT license
* Grisu2 : MIT license
