# C2SourcePawn
Simple C -> SourcePawn translator (SM 1.12 / `newdecls`).

## Usage
0. Copy the whole project into: `<path/to/sourcemod/scripting>`
1. Generate `slim.h` from SourceMod includes:
   - `c-genslim.bat -i .\\include`
2. Write your plugin in C and include `slim.h`:
   - `#include "slim.h"`
3. Translate C -> SourcePawn:
   - `c-translate.bat -i .\\plugin.c -o .\\output.sp`

## Notes
- `c-translate.bat` performs translation in two passes:
  1) generate `.sp`
  2) (optional) run `spcomp` to collect "already defined" symbols, then re-translate with `--ignore-symbols`
- Translate logs are written to: `.\\translate\\` (pass1 and final output).
