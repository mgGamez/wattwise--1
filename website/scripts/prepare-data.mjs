import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const websiteRoot = path.resolve(scriptDirectory, "..");
const repositoryRoot = path.resolve(websiteRoot, "..");
const resultsRoot = path.join(repositoryRoot, "results");

function parseCsv(source) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    const next = source[index + 1];
    if (quoted) {
      if (character === '"' && next === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (field || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }

  const [headers, ...dataRows] = rows;
  return dataRows.map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])),
  );
}

const number = (value) => Number.parseFloat(value);
const boolean = (value) => value.toLowerCase() === "true";

const metrics = JSON.parse(
  await fs.readFile(path.join(resultsRoot, "metrics.json"), "utf8"),
);
const predictionsRaw = parseCsv(
  await fs.readFile(path.join(resultsRoot, "predictions.csv"), "utf8"),
);
const anomaliesRaw = parseCsv(
  await fs.readFile(path.join(resultsRoot, "anomalies.csv"), "utf8"),
);
const failuresRaw = parseCsv(
  await fs.readFile(path.join(resultsRoot, "failure_analysis.csv"), "utf8"),
);

const predictions = predictionsRaw.map((row) => ({
  timestamp: row.timestamp,
  actual: number(row.actual_consumption_wh),
  baseline: number(row.baseline_prediction_wh),
  expected: number(row.predicted_consumption_wh),
  residual: number(row.residual_wh),
  percentAbove: number(row.percentage_above_prediction),
  percentile: number(row.anomaly_percentile),
  isAnomaly: boolean(row.is_anomaly),
}));

const anomalies = anomaliesRaw.map((row) => ({
  timestamp: row.timestamp,
  expected: number(row.predicted_consumption_wh),
  actual: number(row.actual_consumption_wh),
  residual: number(row.residual_wh),
  percentAbove: number(row.percentage_above_prediction),
  percentile: number(row.anomaly_percentile),
}));

const failures = failuresRaw.map((row) => ({
  timestamp: row.timestamp,
  actual: number(row.consumption_wh),
  expected: number(row.predicted_consumption_wh),
  residual: number(row.residual_wh),
  absoluteError: number(row.absolute_error_wh),
  canObserve: row.what_model_can_observe,
  cannotObserve: row.what_model_cannot_observe,
}));

const strongestAnomaly = [...anomalies].sort(
  (first, second) => second.residual - first.residual,
)[0];
const strongestDay = strongestAnomaly.timestamp.slice(0, 10);
const chartData = predictions
  .filter((row) => {
    if (!row.timestamp.startsWith(strongestDay)) return false;
    const hour = Number.parseInt(row.timestamp.slice(11, 13), 10);
    return hour >= 4 && hour <= 12;
  })
  .map((row) => ({ ...row, label: row.timestamp.slice(11, 16) }));

const highNormal = predictions
  .filter((row) => !row.isAnomaly && row.actual >= 300)
  .sort(
    (first, second) =>
      Math.abs(first.residual) - Math.abs(second.residual),
  )[0];
const lowerUnusual = predictions
  .filter((row) => row.isAnomaly && row.actual < highNormal.actual)
  .sort((first, second) => first.actual - second.actual)[0];

const validationModels = Object.entries(metrics.validation_model_comparison).map(
  ([name, values]) => ({
    name,
    mae: values.mae_wh,
    rmse: values.rmse_wh,
    selected: name === metrics.selected_model.name,
  }),
);

const siteData = {
  source: {
    metrics: "results/metrics.json",
    predictions: "results/predictions.csv",
    anomalies: "results/anomalies.csv",
    failures: "results/failure_analysis.csv",
  },
  dataset: metrics.dataset,
  periods: {
    train: metrics.train_period,
    validation: metrics.validation_period,
    test: metrics.test_period,
  },
  validationModels,
  selectedModel: metrics.selected_model,
  finalBaseline: metrics.final_test_baseline,
  finalModel: metrics.final_test_selected_model_results,
  improvement: metrics.percentage_improvement_over_baseline,
  anomalyMethodology: metrics.anomaly_methodology,
  strongestAnomaly,
  anomalies,
  chart: {
    title: `${strongestDay} · 04:00–12:50`,
    rows: chartData,
  },
  context: { highNormal, lowerUnusual },
  failures,
};

const outputDirectory = path.join(websiteRoot, "src", "data");
await fs.mkdir(outputDirectory, { recursive: true });
await fs.writeFile(
  path.join(outputDirectory, "site-data.json"),
  `${JSON.stringify(siteData, null, 2)}\n`,
  "utf8",
);

console.log(
  `Prepared ${anomalies.length} anomalies, ${chartData.length} chart rows, and ${failures.length} failure cases from real ML outputs.`,
);
