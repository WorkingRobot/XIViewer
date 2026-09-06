#!/usr/bin/env bun
// Stands this viewer where a captured game frame was taken from and measures one against the other.
//
//   CHROMIUM=$(...) bun smoke/frame.ts --capture=~/rdcaps/tuli.zip.xml \
//       --level=bg/ex5/02_ykt_y6/twn/y6t1/level/y6t1.lvb --time=14:10 --weather=1 \
//       --crop=150,170,1450,1040 --mask=0,0,2048,70 \
//       --out=smoke/y6t1
//
// The camera and the lens come out of the capture, so the two views are the same view rather than
// two hand-flown ones. The build that drew the frame states its own commit, and a run against a
// build older than the last change to what the wasm is made of fails rather than reporting the
// difference; `--build=<sha>` is how a run against a deliberately older wasm is asked for.

import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

import { Cdp } from "./cdp.ts";

const here = dirname(new URL(import.meta.url).pathname);
const root = resolve(here, "..");

const argv = Bun.argv.slice(2);
const flag = (name: string, fallback: string) =>
    argv.find((one) => one.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const flags = (name: string) =>
    argv.filter((one) => one.startsWith(`--${name}=`)).map((one) => one.slice(name.length + 3));

const home = process.env.HOME ?? "";
const untilde = (path: string) => (path.startsWith("~/") ? join(home, path.slice(2)) : path);

const capture = untilde(flag("capture", ""));
const level = flag("level", "");
const outDir = resolve(root, flag("out", "smoke/frame"));
const dist = resolve(root, "viewer/dist");
const wait = Number(flag("wait", "150000"));
const [WIDTH, HEIGHT] = flag("size", "2400x1200").split("x").map(Number);
const build = flag("build", "");

const sleep = (ms: number) => new Promise((ok) => setTimeout(ok, ms));

// Serves what was just built, so nothing between the build and the browser can hand over an older
// one. The build the frame came from states its own commit either way, and a mismatch is fatal.
function serve() {
    return Bun.serve({
        port: 0,
        fetch(request) {
            const url = new URL(request.url);
            const asked = join(dist, decodeURIComponent(url.pathname));
            if (asked.startsWith(dist) && !asked.endsWith("/") && existsSync(asked)) {
                return new Response(Bun.file(asked));
            }
            return new Response(Bun.file(join(dist, "index.html")), {
                headers: { "content-type": "text/html" },
            });
        },
    });
}

// What the wasm is built from. A commit that only moved the harness leaves the frame valid, so the
// build is checked against the last change to these rather than against HEAD.
const SOURCES = [
    "viewer", ":!viewer/examples",
    "shaders", "shadermerge", "glyphnames", "luadec", "pathlist", "deps",
];

function run(command: string[], where = root) {
    const held = Bun.spawnSync(command, { cwd: where, stdout: "pipe", stderr: "pipe" });
    const out = held.stdout.toString();
    const bad = held.stderr.toString();
    if (held.exitCode !== 0) {
        throw new Error(`${command.join(" ")} failed:\n${bad || out}`);
    }
    return out;
}

async function launch(profile: string) {
    const child = Bun.spawn(
        [
            process.env.CHROMIUM ?? "chromium",
            "--headless",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--enable-unsafe-swiftshader",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            "--mute-audio",
            `--window-size=${WIDTH},${HEIGHT}`,
            `--user-data-dir=${profile}`,
            "--remote-debugging-port=0",
            "about:blank",
        ],
        { stdout: "pipe", stderr: "pipe" },
    );
    const portFile = join(profile, "DevToolsActivePort");
    const deadline = Date.now() + 60_000;
    while (!existsSync(portFile) && Date.now() < deadline) await sleep(200);
    await sleep(300);
    return { child, port: readFileSync(portFile, "utf8").split("\n")[0].trim() };
}

async function click(cdp: Cdp, x: number, y: number) {
    const base = { x, y, button: "left", clickCount: 1, buttons: 1 };
    await cdp.send("Input.dispatchMouseEvent", { ...base, type: "mouseMoved", buttons: 0 });
    await sleep(60);
    await cdp.send("Input.dispatchMouseEvent", { ...base, type: "mousePressed" });
    await sleep(40);
    await cdp.send("Input.dispatchMouseEvent", { ...base, type: "mouseReleased", buttons: 0 });
}

// The paste box is the one way in from outside: the button under it takes no synthetic click, and
// the `TextEdit` applies on Enter instead. Repeated, because a press landing between two frames of
// a loading zone is simply lost, and the box keeps what the last press put in it.
async function paste(cdp: Cdp, text: string) {
    // The scene tab takes a preset off any paste that opens like one, so this hands the page a
    // paste event outright rather than driving the Import menu by pixel.
    const held = JSON.stringify(text);
    await cdp.send("Runtime.evaluate", {
        expression: `(() => {
            const data = new DataTransfer();
            data.setData("text/plain", ${held});
            for (const target of [document, window, ...document.querySelectorAll("canvas")]) {
                target.dispatchEvent(new ClipboardEvent("paste", {
                    clipboardData: data, bubbles: true, cancelable: true,
                }));
            }
        })()`,
    });
    await sleep(4000);
}

async function main() {
    if (!capture) throw new Error("--capture wants a converted RenderDoc capture");
    if (!level) throw new Error("--level wants the .lvb path the capture stood in");
    mkdirSync(outDir, { recursive: true });

    const source = run(["git", "log", "-1", "--format=%H", "--", ...SOURCES]).trim();
    console.log(`tree ${run(["git", "rev-parse", "HEAD"]).trim()}, last drawn by ${source}`);
    if (!argv.includes("--no-build")) {
        console.log("building viewer/dist");
        run(["trunk", "build", "index.html", "--release"], join(root, "viewer"));
    }
    if (!existsSync(join(dist, "index.html"))) {
        throw new Error(`no build at ${dist}`);
    }
    console.log(`dist built ${statSync(join(dist, "index.html")).mtime.toISOString()}`);
    run(["cargo", "build", "-q", "--release", "-p", "framediff"]);
    const rdframe = join(root, "target/release/rdframe");
    console.log(
        run([
            rdframe, capture, `--level=${level}`,
            `--time=${flag("time", "12:00")}`, `--weather=${flag("weather", "1")}`,
            ...flags("camera").map((one) => `--camera=${one}`),
            `--out=${outDir}`,
        ]).trim(),
    );
    const preset = readFileSync(join(outDir, "preset.te3"), "utf8").trim();

    const server = serve();
    const origin = `http://127.0.0.1:${server.port}`;
    const profile = mkdtempSync(join(tmpdir(), "xiviewer-frame-"));
    const { child, port } = await launch(profile);
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const cdp = await Cdp.connect(targets.find((one: any) => one.type === "page").webSocketDebuggerUrl);
    cdp.on("Runtime.exceptionThrown", (p: any) => {
        const held = p.exceptionDetails ?? {};
        console.log(`   !! ${String(held.exception?.description ?? held.text).slice(0, 400)}`);
    });
    cdp.on("Log.entryAdded", (p: any) => {
        const held = p.entry ?? {};
        if (held.level === "error" || /panicked at/.test(held.text ?? "")) {
            console.log(`   !! ${held.source}/${held.level}: ${String(held.text).slice(0, 300)}`);
        }
    });
    let state: any = null;
    try {
        await cdp.send("Runtime.enable");
        await cdp.send("Page.enable");
        await cdp.send("Log.enable");
        await cdp.send("Emulation.setDeviceMetricsOverride", {
            width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: false,
        });
        // The Zones route, not the asset listing: a `.lvb` opened as an asset offers Tree and
        // Sounds and no scene at all, since only that route turns the placed view on.
        await cdp.send("Page.navigate", { url: `${origin}/zones/${level}` });
        await cdp.eval("localStorage.clear()").catch(() => {});
        await sleep(40_000);
        await paste(cdp, preset);

        const began = Date.now();
        let last = "";
        while (Date.now() - began < wait) {
            await sleep(10_000);
            state = await cdp.eval("JSON.parse(window.__frame ?? 'null')").catch(() => null);
            if (!state) continue;
            const now = `drawn ${state.drawn} of ${state.placed}, models ${state.models}, exposure ${state.exposure?.toFixed?.(3)}`;
            if (now !== last) {
                console.log(`   ${Math.round((Date.now() - began) / 1000)}s  ${now}`);
                last = now;
            }
        }
        state = await cdp.eval("JSON.parse(window.__frame ?? 'null')");
        if (!state) throw new Error("the viewer never reported a frame: no scene was drawn");
        writeFileSync(join(outDir, "state.json"), JSON.stringify(state, null, 2));
        const shot = await cdp.send("Page.captureScreenshot", { format: "png" });
        writeFileSync(join(outDir, "window.png"), Buffer.from(shot.data, "base64"));
    } finally {
        cdp.close();
        child.kill();
        await server.stop(true);
        rmSync(profile, { recursive: true, force: true });
    }

    if (!state.clean) {
        console.log(`   the build was made from a dirty tree, so ${state.commit} does not name it`);
    }
    const behind = Bun.spawnSync(["git", "merge-base", "--is-ancestor", source, state.commit], {
        cwd: root,
    }).exitCode !== 0;
    if (behind && !build) {
        throw new Error(
            `the frame was drawn by ${state.commit}, which does not carry ${source}, the last ` +
            "commit to touch what the wasm is built from: rebuild without --no-build",
        );
    }
    const framediff = join(root, "target/release/framediff");
    const args = [
        framediff,
        join(outDir, "frame.png"),
        join(outDir, "window.png"),
        `--capture=${capture}`,
        `--state=${join(outDir, "state.json")}`,
        ...(build ? [`--commit=${build}`] : []),
        `--out=${outDir}`,
        ...flags("crop").map((one) => `--crop=${one}`),
        ...flags("mask").map((one) => `--mask=${one}`),
        ...flags("grid").map((one) => `--grid=${one}`),
    ];
    console.log(run(args));
}

await main();
