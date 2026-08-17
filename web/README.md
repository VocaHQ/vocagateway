# vocahq.github.io/vocagateway

Static landing page for VocaGateway. No build step, no trackers, system fonts.

```bash
python3 -m http.server 4173 --directory .
node --test tests/site.test.mjs
```

GitHub Pages deploys this directory from `main`. After the first merge, set
the repository Pages source to **GitHub Actions**.
