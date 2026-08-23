# vocagateway.vocahq.com

Static landing page for VocaGateway. No build step, no trackers, system fonts.

```bash
python3 -m http.server 4173 --directory .
node --test tests/site.test.mjs
```

GitHub Pages deploys this directory from `main`. After the first merge, set
the repository Pages source to **GitHub Actions** and point
`vocagateway.vocahq.com` at the Pages host.

The Open Graph card is `assets/og-image.png` (1200×630), drawn from
`assets/og/src/og-default.html` in the same paper language as VocaMac,
VocaPhone, and VocaHQ. The right side is the charcoal WebUI from this page,
not a generic logo. Serve the site and open `/assets/og/src/preview.html` to
proof it at native size.
