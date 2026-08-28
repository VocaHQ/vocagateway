import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const siteRoot = fileURLToPath(new URL("..", import.meta.url));
const html = readFileSync(join(siteRoot, "index.html"), "utf8");
const css = readFileSync(join(siteRoot, "styles.css"), "utf8");
const script = readFileSync(join(siteRoot, "script.js"), "utf8");

function pngDimensions(buffer) {
  assert.equal(buffer.toString("ascii", 1, 4), "PNG", "asset must be a PNG");
  return {
    width: buffer.readUInt32BE(16),
    height: buffer.readUInt32BE(20),
  };
}

test("page has one title and landmark structure", () => {
  assert.match(html, /<title>VocaGateway: self-hosted speech-to-text on your hardware<\/title>/);
  assert.equal((html.match(/<h1\b/g) || []).length, 1);
  assert.match(html, /<main id="main-content">/);
  assert.match(html, /<nav[^>]+aria-label="Main navigation"/);
  assert.match(html, /class="skip-link"/);
});

test("in-page links have matching section ids", () => {
  const anchors = [...html.matchAll(/href="#([\w-]+)"/g)].map((match) => match[1]);
  assert.ok(anchors.length > 0);
  for (const anchor of anchors) {
    assert.match(html, new RegExp(`id="${anchor}"`), `Missing #${anchor}`);
  }
});

test("document ids are unique", () => {
  const ids = [...html.matchAll(/\sid="([\w-]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
});

test("product truth stays scoped to Beta self-hosted infrastructure", () => {
  assert.match(html, /your hardware, shared speech-to-text/i);
  assert.match(html, /\bbeta\b/i);
  assert.doesNotMatch(html, /\bearly\b/i);
  assert.doesNotMatch(html, /\balpha\b/i);
  assert.match(html, /not on-device/i);
  assert.match(html, /audio leaves/i);
  assert.match(html, /trusted LAN/i);
  assert.match(html, /Tailscale|encrypted/i);
  assert.match(html, /HTTPS/i);
  assert.match(html, /8765/);
  assert.match(html, /public internet/i);
  assert.match(html, /no Voca account/i);
  assert.match(html, /AGPL-3\.0/);
  assert.match(html, /Ready for dictation/);
  assert.match(html, /https:\/\/github\.com\/VocaHQ\/vocagateway\/releases\/tag\/v0\.1\.0/);
  assert.match(html, /class="button button-primary" href="https:\/\/github\.com\/VocaHQ\/vocagateway\/releases\/tag\/v0\.1\.0"/);
  assert.match(html, /no packaged installer/i);
  assert.doesNotMatch(html, /download the beta \.exe/i);
  assert.doesNotMatch(html, /https?:\/\/vocagateway\.com\b/i);
  assert.doesNotMatch(html, /100% offline/i);
  assert.doesNotMatch(html, /free forever/i);
  assert.doesNotMatch(html, /AI-powered/i);
  assert.doesNotMatch(html, /military-grade/i);
  assert.doesNotMatch(html, /VocaServer/);
  assert.doesNotMatch(html, /googletagmanager|gtag\(|G-SHWKRJMCEN/i);
});

test("names the primary CLI vocagateway and keeps deprecated aliases honest", () => {
  assert.match(html, /uv run vocagateway/);
  assert.match(html, /vocagateway<\/code> is the CLI/);
  assert.match(html, /Deprecated[\s\S]*vocaphone-server/);
  assert.doesNotMatch(html, /product is vocaphone-server/i);
  assert.doesNotMatch(html, /leftover CLI entry point/);
});

test("source, org, and family products are linked", () => {
  assert.match(html, /href="https:\/\/github\.com\/VocaHQ\/vocagateway"/);
  assert.match(html, /href="https:\/\/github\.com\/VocaHQ"/);
  assert.match(html, /href="https:\/\/vocahq\.com\/"/);
  assert.match(html, /href="https:\/\/vocalinux\.com\/"/);
  assert.match(html, /href="https:\/\/vocamac\.com\/"/);
  assert.match(html, /href="https:\/\/vocawin\.com\/"/);
  assert.match(html, /href="https:\/\/github\.com\/VocaHQ\/vocawin\/tree\/v0\.1\.0-beta\.1"/);
  assert.doesNotMatch(html, /href="https:\/\/github\.com\/VocaHQ\/vocawin\/releases"/);
  assert.match(html, /href="https:\/\/vocaphone\.vocahq\.com\/"/);
  assert.match(html, /href="https:\/\/github\.com\/VocaHQ\/vocaphone"/);
  assert.match(html, /href="https:\/\/discord\.gg\/t6muquAJbm"/);
  assert.match(html, /href="https:\/\/x\.com\/vocahq"/);
});

test("VocaWin family card is Beta with honest unsigned-tag copy", () => {
  const winCard = html.match(
    /<article class="eco-card reveal">\s*<div class="eco-head">\s*<img src="assets\/icons\/windows\.svg"[\s\S]*?<\/article>/,
  );
  assert.ok(winCard, "VocaWin eco-card is present");
  assert.match(winCard[0], /<small>beta<\/small>/);
  assert.match(winCard[0], /Unsigned/);
  assert.match(winCard[0], /v0\.1\.0-beta\.1/);
  assert.match(winCard[0], /SmartScreen/);
  assert.match(winCard[0], /not a store listing/i);
  assert.match(winCard[0], /does\s+not expose a gateway mode today/);
  assert.doesNotMatch(winCard[0], /coming soon/i);
  assert.doesNotMatch(winCard[0], /no public installer/i);
  assert.doesNotMatch(winCard[0], /available now/i);
  assert.doesNotMatch(winCard[0], /\/releases/);
  assert.doesNotMatch(winCard[0], /GitHub Releases/);
});

test("VocaPhone family card names public TestFlight, not source-only iPhone", () => {
  const phoneCard = html.match(
    /<article class="eco-card reveal">\s*<div class="eco-head">\s*<img src="assets\/icons\/android\.svg"[\s\S]*?<\/article>/,
  );
  assert.ok(phoneCard, "VocaPhone eco-card is present");
  assert.match(phoneCard[0], /<h3>VocaPhone<\/h3>/);
  assert.match(phoneCard[0], /<small>beta \/ testflight<\/small>/);
  assert.match(phoneCard[0], /Android has a public beta/);
  assert.match(phoneCard[0], /href="https:\/\/testflight\.apple\.com\/join\/wd85wQ3W"/);
  assert.match(phoneCard[0], /eco-links[\s\S]*TestFlight/);
  assert.match(phoneCard[0], /1,?000 seats/);
  assert.match(phoneCard[0], /iOS 17\+ source build/);
  assert.match(phoneCard[0], /href="https:\/\/vocaphone\.vocahq\.com\/"/);
  assert.match(phoneCard[0], /href="https:\/\/github\.com\/VocaHQ\/vocaphone"/);
  assert.match(phoneCard[0], /on-device is the default path/i);
  assert.match(phoneCard[0], /gateway is optional/i);
  assert.doesNotMatch(phoneCard[0], /beta \/ source build/);
  assert.doesNotMatch(phoneCard[0], /currently needs an iOS 17\+ source build/);
});

test("hero shows pairing, readiness, and charcoal dashboard chrome", () => {
  assert.match(html, /Ready for dictation/);
  assert.match(html, /Pair phone/);
  assert.match(html, /class="hero-demo dash/);
  assert.match(css, /--charcoal:\s*#1c1e1c/);
  assert.doesNotMatch(html, /traffic-lights/);
  assert.doesNotMatch(html, /class="win-captions"/);
});

test("all local image assets exist", () => {
  const localImages = [...html.matchAll(/(?:src|href)="((?:assets|favicon)[^"]+)"/g)].map(
    (match) => match[1],
  );
  assert.ok(localImages.includes("assets/brand/voca-logo.svg"));
  assert.ok(localImages.includes("assets/brand/vocagateway/vocagateway-tower.svg"));
  assert.match(html, /class="hero-mark" src="assets\/brand\/vocagateway\/vocagateway-tower\.svg"/);
  assert.doesNotMatch(html, /class="hero-mark" src="assets\/brand\/voca-mark\.svg"/);
  for (const asset of localImages) {
    assert.ok(existsSync(join(siteRoot, asset)), `Missing ${asset}`);
  }
});

test("production metadata is complete", () => {
  assert.match(html, /rel="canonical" href="https:\/\/vocagateway\.vocahq\.com\/"/);
  assert.match(html, /property="og:url" content="https:\/\/vocagateway\.vocahq\.com\/"/);
  assert.match(
    html,
    /property="og:image" content="https:\/\/vocagateway\.vocahq\.com\/assets\/og-image\.png"/,
  );
  assert.match(html, /property="og:image:width" content="1200"/);
  assert.match(html, /property="og:image:height" content="630"/);
  assert.match(html, /name="twitter:card" content="summary_large_image"/);
  assert.match(
    html,
    /name="twitter:image" content="https:\/\/vocagateway\.vocahq\.com\/assets\/og-image\.png"/,
  );
  assert.doesNotMatch(html, /github\.io\/vocagateway/);
  assert.equal(
    readFileSync(join(siteRoot, "CNAME"), "utf8").trim(),
    "vocagateway.vocahq.com",
  );
  for (const asset of [
    "assets/og-image.png",
    "assets/og/src/og-default.html",
    "assets/og/src/preview.html",
    "assets/brand/voca-logo.svg",
    "assets/brand/voca-mark.svg",
    "assets/brand/vocagateway/vocagateway-tower.svg",
    "assets/brand/voca-logo-512.png",
    "assets/paper-dots.svg",
    "favicon.svg",
    "robots.txt",
    "sitemap.xml",
    "site.webmanifest",
    "CNAME",
  ]) {
    assert.ok(existsSync(join(siteRoot, asset)), `Missing ${asset}`);
  }

  const ogImage = readFileSync(join(siteRoot, "assets/og-image.png"));
  assert.deepEqual(pngDimensions(ogImage), { width: 1200, height: 630 });
});

test("Open Graph card follows the Voca paper language", () => {
  const ogSource = readFileSync(join(siteRoot, "assets/og/src/og-default.html"), "utf8");
  assert.match(ogSource, /--paper:\s*#f4f1e8/);
  assert.match(ogSource, /--ink:\s*#14231c/);
  assert.match(ogSource, /--brand:\s*#0f6b57/);
  assert.match(ogSource, /--charcoal:\s*#1c1e1c/);
  assert.match(ogSource, /not on-device/i);
  assert.match(ogSource, /Ready for dictation/);
  assert.match(ogSource, /\bbeta\b/i);
  assert.doesNotMatch(ogSource, /\bearly\b/i);
  assert.doesNotMatch(ogSource, /traffic-lights/);
  const bannedFunction = ["linear-" + "gradient", "radial-" + "gradient", "conic-" + "gradient"];
  for (const token of bannedFunction) {
    assert.ok(!ogSource.includes(token), `Unexpected ${token} in OG source`);
  }
});

test("visual treatment stays flat", () => {
  const bannedFunction = ["linear-" + "gradient", "radial-" + "gradient", "conic-" + "gradient"];
  for (const token of bannedFunction) {
    assert.ok(!css.includes(token), `Unexpected ${token}`);
  }
});

test("motion has a reduced-motion fallback", () => {
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
  assert.match(script, /setTimeout\([\s\S]*revealNodes[\s\S]*is-visible/);
});

test("mobile navigation can be dismissed with the keyboard", () => {
  assert.match(script, /event\.key === "Escape"/);
  assert.match(script, /closeNavigation\(\{ returnFocus: true \}\)/);
});

test("landing vendors official WebUI captures and does not treat the pair PNG as a live QR", () => {
  const overview = "assets/brand/vocagateway/vocagateway-webui-overview-ready.png";
  const pair = "assets/brand/vocagateway/vocagateway-webui-pair-qr.png";
  for (const asset of [overview, pair]) {
    const fullPath = join(siteRoot, asset);
    assert.ok(existsSync(fullPath), `Missing ${asset}`);
    pngDimensions(readFileSync(fullPath));
  }
  assert.match(html, /src="assets\/brand\/vocagateway\/vocagateway-webui-overview-ready\.png"/);
  assert.match(html, /src="assets\/brand\/vocagateway\/vocagateway-webui-pair-qr\.png"/);
  assert.match(html, /Ready for dictation/);
  assert.doesNotMatch(html, /raw\.githubusercontent\.com/);

  const pairFigure = html.match(
    /<figure class="capture-figure reveal">\s*<img\s+src="assets\/brand\/vocagateway\/vocagateway-webui-pair-qr\.png"[\s\S]*?<\/figure>/,
  );
  assert.ok(pairFigure, "pair capture figure is present");
  assert.match(pairFigure[0], /stopped local host/i);
  assert.match(pairFigure[0], /not a live token/i);
  assert.match(pairFigure[0], /scan the QR in your own WebUI/);
  assert.doesNotMatch(pairFigure[0], /scan this (image|png|qr|code)/i);
  assert.doesNotMatch(pairFigure[0], /scan this pairing/i);
  assert.doesNotMatch(pairFigure[0], /point (your )?phone at this/i);
});
