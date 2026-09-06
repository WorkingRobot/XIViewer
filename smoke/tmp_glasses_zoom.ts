#!/usr/bin/env bun
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Cdp } from "./cdp.ts";

const root = resolve(import.meta.dir, "..");
const dist = resolve(root, "viewer/dist");
const outDir = join(root, "smoke", "glasses_probe3");
const sleep = (ms: number) => new Promise((ok) => setTimeout(ok, ms));

function serve() {
    return Bun.serve({
        port: 9090,
        async fetch(request) {
            const asked = join(dist, decodeURIComponent(new URL(request.url).pathname));
            if (asked.startsWith(dist) && !asked.endsWith("/")) {
                const file = Bun.file(asked);
                if (await file.exists()) return new Response(file);
            }
            return new Response(Bun.file(join(dist, "index.html")), {
                headers: { "content-type": "text/html; charset=utf-8" },
            });
        },
    });
}

async function launch(profile: string) {
    const child = Bun.spawn(
        [
            process.env.CHROMIUM ?? "chromium",
            "--headless", "--no-sandbox", "--disable-dev-shm-usage",
            "--enable-unsafe-swiftshader", "--no-first-run", "--no-default-browser-check",
            "--hide-scrollbars", "--mute-audio", "--window-size=1600,1000",
            `--user-data-dir=${profile}`, "--remote-debugging-port=0", "about:blank",
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
    await sleep(1500);
    await cdp.send("Input.dispatchMouseEvent", { ...base, type: "mousePressed" });
    await sleep(400);
    await cdp.send("Input.dispatchMouseEvent", { ...base, type: "mouseReleased", buttons: 0 });
    await sleep(1200);
}

async function wheel(cdp: Cdp, x: number, y: number, by: number, times: number) {
    await cdp.send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y, buttons: 0 });
    await sleep(200);
    for (let i = 0; i < times; i++) {
        await cdp.send("Input.dispatchMouseEvent", { type: "mouseWheel", x, y, deltaX: 0, deltaY: by });
        await sleep(120);
    }
    await sleep(500);
}

async function shot(cdp: Cdp, name: string) {
    const held = await cdp.send("Page.captureScreenshot", { format: "png" });
    writeFileSync(join(outDir, `${name}.png`), Buffer.from(held.data, "base64"));
    console.log(`   ${name}.png`);
}

async function waitFor(what: string, timeoutMs: number, check: () => boolean) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (check()) return;
        await sleep(200);
    }
    throw new Error(`timed out waiting for ${what}`);
}

async function clickUntil(cdp: Cdp, point: { x: number; y: number }, read: () => string, reset: () => void, wants: string, what: string) {
    let last = "";
    for (let attempt = 0; attempt < 6; attempt++) {
        reset();
        await click(cdp, point.x, point.y);
        try {
            await waitFor(what, 4_000, () => read() === wants);
            return;
        } catch {
            last = read();
        }
    }
    throw new Error(`click at (${point.x},${point.y}) never landed on ${what} (last: ${last})`);
}

async function main() {
    mkdirSync(outDir, { recursive: true });
    const server = serve();
    const origin = `http://127.0.0.1:${server.port}`;
    const profile = mkdtempSync(join(tmpdir(), "xiviewer-glasses-zoom-"));
    const { child, port } = await launch(profile);
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const cdp = await Cdp.connect(targets.find((one: any) => one.type === "page").webSocketDebuggerUrl);

    let picking = "";
    let chose = "";
    let pieced = 0;
    const text = (a: any) => (a?.value !== undefined ? String(a.value) : (a?.description ?? a?.type ?? ""));
    cdp.on("Runtime.consoleAPICalled", (p: any) => {
        const line = p.args.map(text).join(" ");
        if (!line.includes("character")) return;
        console.log(`   | ${line}`);
        const pieces = line.match(/character: (\d+) pieces to wear/);
        if (pieces) pieced = Number(pieces[1]);
        const opened = line.match(/character: picking (\w+)/);
        if (opened) picking = opened[1];
        const picked = line.match(/character: chose .* for (\w+)/);
        if (picked) chose = picked[1];
    });
    cdp.on("Log.entryAdded", (p: any) => {
        const held = p.entry ?? {};
        if (held.level === "error" || /panicked at/.test(held.text ?? "")) {
            console.log(`   !! ${held.source}/${held.level}: ${String(held.text).slice(0, 400)}`);
        }
    });
    await cdp.send("Runtime.enable");
    await cdp.send("Page.enable");
    await cdp.send("Log.enable");
    await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });

    try {
        await cdp.send("Page.navigate", { url: `${origin}/character` });
        await waitFor("pieces to load", 180_000, () => pieced > 0);
        await sleep(4000);

        const FACEWEAR = { x: 90, y: 676 };
        const ITEM = { x: 140, y: 815 };
        await clickUntil(cdp, FACEWEAR, () => picking, () => (picking = ""), "Facewear", "Facewear's picker");
        await sleep(2000);
        await clickUntil(cdp, ITEM, () => chose, () => (chose = ""), "Facewear", "an item chosen for Facewear");
        await sleep(6000);
        // Zoom the camera onto the face.
        await wheel(cdp, 950, 300, -120, 14);
        await sleep(1500);
        await shot(cdp, "00-zoomed");
        console.log("SUCCESS");
    } finally {
        cdp.close();
        child.kill();
        server.stop(true);
        rmSync(profile, { recursive: true, force: true });
    }
}

await main();
