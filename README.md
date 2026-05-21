# Bob L'eponge Pose Game

Static Teachable Machine pose game ready for GitHub + Vercel deploy.

## What runs

- `index.html` is the site entrypoint
- `static/app.js` runs webcam + Teachable Machine inference in the browser
- `static/styles.css` contains the full UI theme
- `photos/` contains meme images matched by filename to model class names

## Filename rule

Model class names must match meme filenames without extension.

Examples:

- `angry` -> `photos/angry.png`
- `kurnaz` -> `photos/kurnaz.png`
- `perfect` -> `photos/perfect.png`

## Local preview

You do not need FastAPI.

Use any static server, for example:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Vercel deploy

1. Push this repo to GitHub
2. In Vercel click `Connect Git Repository`
3. Select this repo
4. Deploy with default settings

No build command is required.

## Notes

- The app first tries the Teachable Machine cloud model:
  `https://teachablemachine.withgoogle.com/models/tTYdfh6E2/`
- If you later want a local model export, place it in `static/my_model/`
- Webcam requires HTTPS in production, which Vercel provides automatically
