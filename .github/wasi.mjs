// Runs a wasm32-wasip1 binary under the WASI preview 1 of Node.js, as the cargo
// runner of that target: node wasi.mjs <binary.wasm> [arguments...]
import { readFileSync } from 'node:fs';
import { argv, env, exit } from 'node:process';
import { WASI } from 'node:wasi';

const wasi = new WASI({
  version: 'preview1',
  args: argv.slice(2),
  env,
  returnOnExit: true,
});
const module = new WebAssembly.Module(readFileSync(argv[2]));
const instance = new WebAssembly.Instance(module, wasi.getImportObject());
exit(wasi.start(instance));
