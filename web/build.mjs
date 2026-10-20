import { build } from "esbuild";
import { mkdir, cp, rm, readdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = dirname(fileURLToPath(import.meta.url));
await rm(resolve(root, "dist"), { recursive: true, force: true });
await mkdir(resolve(root, "dist/assets/ort"), { recursive: true });
await cp(resolve(root, "public"), resolve(root, "dist"), { recursive: true });
await cp(resolve(root, "src/index.html"), resolve(root, "dist/index.html"));
await cp(
  resolve(root, "src/style.css"),
  resolve(root, "dist/assets/style.css"),
);
await cp(resolve(root, "src/app.js"), resolve(root, "dist/assets/app.js"));
await build({
  entryPoints: [resolve(root, "src/worker.js")],
  outfile: resolve(root, "dist/assets/assistant-worker.js"),
  bundle: true,
  platform: "browser",
  format: "esm",
  target: ["es2022"],
  minify: true,
});
const ort = resolve(root, "node_modules/onnxruntime-web/dist");
for (const name of await readdir(ort)) {
  if (
    name.startsWith("ort-wasm-") &&
    (name.endsWith(".wasm") || name.endsWith(".mjs"))
  )
    await cp(resolve(ort, name), resolve(root, "dist/assets/ort", name));
}
console.log("Built static browser application.");
