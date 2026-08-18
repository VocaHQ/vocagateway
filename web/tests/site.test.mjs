import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const siteRoot = fileURLToPath(new URL("..", import.meta.url));
const html = readFileSync(join(siteRoot, "index.html"), "utf8");
const css = readFileSync(join(siteRoot, "styles.css"), "utf8");
const script = readFileSync(join(siteRoot, "script.js"), "utf8");

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

test("product truth stays scoped to Early self-hosted infrastructure", () => {
  assert.match(html, /your hardware, shared speech-to-text/i);
  assert.match(html, /\bearly\b/i);
  assert.match(html, /not on-device/i);
  assert.match(html, /audio leaves/i);
  assert.match(html, /trusted LAN/i);
  assert.match(html, /Tailscale|encrypted/i);
  assert.match(html, /HTTPS/i);
  assert.match(html, /8765/);
  assert.match(html, /public internet/i);
  assert.match(html, /no Voca account/i);
  assert.match(html, /AGPL-3\.0/);
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
  assert.match(html, /href="https:\/\/vocaphone\.vocahq\.com\/"/);
  assert.match(html, /href="https:\/\/github\.com\/VocaHQ\/vocaphone"/);
  assert.match(html, /href="https:\/\/discord\.gg\/UMJduhcqn"/);
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
  assert.ok(localImages.includes("assets/brand/voca-mark.svg"));
  for (const asset of localImages) {
    assert.ok(existsSync(join(siteRoot, asset)), `Missing ${asset}`);
  }
});

test("production metadata is complete", () => {
  assert.match(html, /rel="canonical" href="https:\/\/vocagateway\.vocahq\.com\/"/);
  assert.match(html, /property="og:url" content="https:\/\/vocagateway\.vocahq\.com\/"/);
  assert.match(html, /name="twitter:card" content="summary"/);
  assert.doesNotMatch(html, /github\.io\/vocagateway/);
  assert.equal(
    readFileSync(join(siteRoot, "CNAME"), "utf8").trim(),
    "vocagateway.vocahq.com",
  );
  for (const asset of [
    "assets/brand/voca-logo.svg",
    "assets/brand/voca-mark.svg",
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
