# Showcase Site

Static one-page showcase for the FYP. Pure HTML + CSS + vanilla JS — no build step, no framework.

## Preview locally

Just open `index.html` in a browser, or from this directory:

```bash
python -m http.server 8080
```

Then visit http://localhost:8080.

## Placeholders to fill in

Search `index.html` for `#REPO_URL_PLACEHOLDER` and replace all occurrences with your
public GitHub repo URL (e.g. `https://github.com/folosiye/FYP-NIDS-ML`).

### Demo video

Record a short screen capture (30–60s) of the demo simulator and Flask dashboard.
Save it as `assets/demo.mp4` and then in `index.html`:

1. Delete the `<div class="video-placeholder">…</div>` block.
2. Uncomment the `<video>` block directly below it.

### Downloads

Drop these files into `downloads/`:

- `report.pdf` — your project report
- `poster.pdf` — A1 showcase poster
- `presentation.pdf` — viva slides

If you don't have a poster or slides yet, delete those two cards from `index.html` in
the Downloads section.

## Deploy

- **GitHub Pages:** push `site/` to a repo, point Pages at `/site` (or rename to `/docs`).
- **SETU showcase path:** upload the contents of `site/` to the assigned directory.
- **Netlify / Vercel:** drag-and-drop the `site/` folder.

## Structure

```
site/
├── index.html          # single page, all sections
├── assets/
│   ├── style.css       # Tokyo-Night-ish theme
│   ├── script.js       # active-nav + reveal-on-scroll
│   └── figures/        # copied from ../figures/
└── downloads/          # report.pdf, poster.pdf, etc.
```
