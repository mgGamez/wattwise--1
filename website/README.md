# WATTWISE website

Static Next.js presentation site powered by the repository's generated ML
results. No API, Python backend, or paid service is required at runtime.

## Run locally

```bash
cd website
npm install
npm run dev
```

Open <http://localhost:3000>.

## Production build

```bash
cd website
npm run build
```

The static export is written to `website/out/`.

Both commands run `scripts/prepare-data.mjs` first. It rebuilds
`src/data/site-data.json` from:

- `../results/metrics.json`
- `../results/predictions.csv`
- `../results/anomalies.csv`
- `../results/failure_analysis.csv`

