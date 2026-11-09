# IAAN project page

This site is adapted from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).
The actual page is [`index.html`](index.html), with local styles in `static/css/`.

## View locally

```bash
cd docs
python -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000).

## GitHub Pages

The repository includes an automatic publishing workflow:
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml).

1. Go to **Repository Settings → Pages**.
2. Under **Build and deployment**, choose **GitHub Actions** as the source.
3. Open the **Actions** tab and inspect or manually run **Deploy IAAN project page**.
4. When the workflow succeeds, verify the live URL at
   `https://estherhan1.github.io/IAAN/`.

**Deployment is not yet verified.** The repository is currently private.
Pages publishing from a private repository may require a paid GitHub plan,
and configuring an accessible public Pages URL can make the page content
public even while the underlying repository remains private. Do not
make the repository public solely to publish the page without considering
what else it exposes.

## Credit

Adapted from the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
under its CC BY-SA 4.0 license; Bulma uses MIT.
