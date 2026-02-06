# C2SourcePawn
Simple C → SourcePawn translator (SM 1.12 / `newdecls`).

## Usage
0. Copy the whole repo into: `<path/to/sourcemod/scripting>`
1. Generate `slim.h` from SourceMod includes:
   - `c-genslim.bat -i .\\include`
2. Write your plugin in C and include `slim.h`:
   - `#include "slim.h"`
3. Translate and compile:
   - `c-compile.bat -i .\\plugin.c -o .\\output.sp`

## Notes
- `c-compile.bat` runs translation in two passes:
  1) generate `.sp`  
  2) compile with `spcomp` → collect “already defined” symbols → re-translate with `--ignore-symbols`
- Build logs are written to: `.\\build\\` (pass1 and final compile output).
