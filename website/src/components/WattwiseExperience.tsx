"use client";

import {
  AnimatePresence,
  motion,
  useInView,
  useMotionValue,
  useSpring,
  useTransform,
} from "framer-motion";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useEffect, useMemo, useRef, useState } from "react";
import siteData from "@/data/site-data.json";

type Anomaly = (typeof siteData.anomalies)[number];
type Prediction = (typeof siteData.chart.rows)[number];

const formatNumber = (value: number, digits = 1) =>
  new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);

const formatTimestamp = (timestamp: string) => {
  const [date, time] = timestamp.split(" ");
  const [year, month, day] = date.split("-").map(Number);
  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  return `${day} ${months[month - 1]} ${year} · ${time.slice(0, 5)}`;
};

function Reveal({
  children,
  className = "",
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.16 }}
      transition={{ duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

function CountUp({
  value,
  digits = 0,
  suffix = "",
}: {
  value: number;
  digits?: number;
  suffix?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  const motionValue = useMotionValue(0);
  const spring = useSpring(motionValue, { stiffness: 80, damping: 22 });
  const display = useTransform(spring, (latest) =>
    `${formatNumber(latest, digits)}${suffix}`,
  );

  useEffect(() => {
    if (isInView) motionValue.set(value);
  }, [isInView, motionValue, value]);

  return <motion.span ref={ref}>{display}</motion.span>;
}

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description?: string;
}) {
  return (
    <Reveal className="section-heading">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      {description ? <p className="section-description">{description}</p> : null}
    </Reveal>
  );
}

function SignalGraphic() {
  return (
    <div className="signal-graphic" aria-hidden="true">
      <div className="signal-grid" />
      <div className="signal-label signal-label-top">EXPECTED</div>
      <div className="signal-label signal-label-bottom">ACTUAL</div>
      <svg viewBox="0 0 700 360" preserveAspectRatio="none">
        <motion.path
          d="M0,226 C70,214 90,178 145,189 C210,202 248,235 310,215 C360,198 395,204 434,182 C475,160 493,201 540,188 C598,171 622,159 700,172"
          fill="none"
          stroke="rgba(243,245,237,.42)"
          strokeWidth="3"
          strokeDasharray="7 10"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.2, delay: 0.25 }}
        />
        <motion.path
          d="M0,240 C72,218 95,194 148,202 C214,213 249,245 312,223 C363,204 402,210 438,187 C470,168 490,212 527,196 C555,185 565,72 596,58 C625,44 654,172 700,181"
          fill="none"
          stroke="#8cff5b"
          strokeWidth="5"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.25, delay: 0.4 }}
        />
        <motion.circle
          cx="596"
          cy="58"
          r="9"
          fill="#8cff5b"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 1.4, type: "spring" }}
        />
        <circle cx="596" cy="58" r="22" fill="none" stroke="rgba(140,255,91,.25)" />
      </svg>
      <div className="signal-callout">
        <span>unexpected</span>
        <strong>+{formatNumber(siteData.strongestAnomaly.residual, 1)} Wh</strong>
      </div>
    </div>
  );
}

function MetricTile({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AnomalyCard({ anomaly }: { anomaly: Anomaly }) {
  return (
    <motion.article
      className="explorer-card"
      key={anomaly.timestamp}
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -12 }}
      transition={{ duration: 0.22 }}
    >
      <div className="explorer-card-head">
        <div>
          <p className="eyebrow">Unexpected consumption</p>
          <h3>{formatTimestamp(anomaly.timestamp)}</h3>
        </div>
        <div className="percentile-pill">P{formatNumber(anomaly.percentile, 1)}</div>
      </div>
      <div className="explorer-metrics">
        <MetricTile label="Actual" value={`${formatNumber(anomaly.actual)} Wh`} />
        <MetricTile label="Expected" value={`${formatNumber(anomaly.expected)} Wh`} />
        <MetricTile label="Residual" value={`+${formatNumber(anomaly.residual)} Wh`} />
        <MetricTile
          label="Above expected"
          value={`+${formatNumber(anomaly.percentAbove)}%`}
        />
      </div>
      <p className="fine-print">Wh per 10-minute interval · percentile versus positive validation residuals</p>
    </motion.article>
  );
}

export function WattwiseExperience() {
  const [anomalyIndex, setAnomalyIndex] = useState(0);
  const anomaly = siteData.anomalies[anomalyIndex];
  const chartRows = useMemo(
    () =>
      siteData.chart.rows.map((row) => ({
        ...row,
        anomalyActual: row.isAnomaly ? row.actual : null,
      })),
    [],
  );

  const moveAnomaly = (direction: number) => {
    setAnomalyIndex((current) =>
      (current + direction + siteData.anomalies.length) % siteData.anomalies.length,
    );
  };

  return (
    <main>
      <header className="site-nav">
        <a className="brand" href="#top" aria-label="WATTWISE home">
          <span className="brand-mark">W</span>
          WATTWISE
        </a>
        <nav aria-label="Primary navigation">
          <a href="#how">How it works</a>
          <a href="#evidence">Evidence</a>
          <a className="nav-cta" href="#explore">Explore anomalies</a>
        </nav>
      </header>

      <section className="hero" id="top">
        <motion.div
          className="hero-copy"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65 }}
        >
          <p className="eyebrow light">AI + CLIMATE · REAL MODEL OUTPUTS</p>
          <h1>
            Your building is talking.
            <span>WATTWISE listens.</span>
          </h1>
          <p className="hero-lede">
            AI that learns normal electricity patterns and finds what doesn&apos;t belong.
          </p>
          <div className="hero-actions">
            <a className="primary-button" href="#anomaly">Explore the AI</a>
            <span>Built from {siteData.dataset.modeling_rows.toLocaleString()} real observations</span>
          </div>
        </motion.div>
        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.15 }}
        >
          <SignalGraphic />
        </motion.div>
      </section>

      <section className="problem section-shell">
        <Reveal className="problem-grid">
          <p className="section-index">01 / THE PROBLEM</p>
          <h2>Knowing how much energy you used isn&apos;t enough.</h2>
          <div className="problem-copy">
            <p>Traditional dashboards show consumption.</p>
            <p className="question">Was this amount normal for this building at this moment?</p>
          </div>
        </Reveal>
      </section>

      <section className="anomaly-proof section-shell" id="anomaly">
        <SectionHeading
          eyebrow="02 / A REAL SIGNAL"
          title="The gap is the story."
          description="The strongest positive residual in the final unseen test period."
        />
        <Reveal className="proof-card">
          <div className="proof-context">
            <span className="status-dot" /> Unexpected consumption
            <h3>{formatTimestamp(siteData.strongestAnomaly.timestamp)}</h3>
            <p>Wh per 10-minute interval</p>
          </div>
          <div className="proof-numbers">
            <div>
              <span>Actual</span>
              <strong><CountUp value={siteData.strongestAnomaly.actual} /> Wh</strong>
            </div>
            <div>
              <span>AI expected</span>
              <strong><CountUp value={siteData.strongestAnomaly.expected} digits={1} /> Wh</strong>
            </div>
            <div className="accent-number">
              <span>Residual</span>
              <strong>+<CountUp value={siteData.strongestAnomaly.residual} digits={1} /> Wh</strong>
            </div>
          </div>
          <div className="proof-footer">
            <span><b>+{formatNumber(siteData.strongestAnomaly.percentAbove, 1)}%</b> above expected</span>
            <span><b>{formatNumber(siteData.strongestAnomaly.percentile, 2)}th</b> calibration percentile</span>
            <span>Requires investigation—not proof of waste.</span>
          </div>
        </Reveal>
      </section>

      <section className="how section-shell" id="how">
        <SectionHeading
          eyebrow="03 / HOW THE AI WORKS"
          title="From a stream of readings to one useful question."
        />
        <Reveal className="ai-flow">
          <div className="input-stack">
            {[
              "Historical electricity use",
              "Time patterns",
              "Recent consumption",
              "Environmental features",
            ].map((label, index) => (
              <div className="flow-input" key={label}>
                <span>0{index + 1}</span>{label}
              </div>
            ))}
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-model"><span>ML model</span><strong>{siteData.selectedModel.name}</strong></div>
          <div className="flow-arrow">→</div>
          <div className="flow-output">
            <div>Expected consumption</div>
            <span>versus</span>
            <div>Actual consumption</div>
            <b>↓</b>
            <strong>Anomaly detection</strong>
          </div>
        </Reveal>
        <Reveal className="formula">
          <span>Residual</span><b>=</b><span>Actual</span><b>−</b><span>Predicted</span>
        </Reveal>
      </section>

      <section className="performance section-shell" id="evidence">
        <SectionHeading
          eyebrow="04 / MODEL PERFORMANCE"
          title="Selected on validation. Judged once on test."
          description="Lower MAE and RMSE are better. Model selection used validation RMSE, never final-test results."
        />
        <div className="validation-grid">
          {siteData.validationModels.map((model, index) => (
            <Reveal key={model.name} delay={index * 0.06} className={`model-card ${model.selected ? "selected" : ""}`}>
              <div className="model-card-head">
                <span>{model.selected ? "SELECTED" : "VALIDATION"}</span>
                <h3>{model.name}</h3>
              </div>
              <div className="model-metrics">
                <div><span>MAE</span><b>{formatNumber(model.mae, 2)}</b></div>
                <div><span>RMSE</span><b>{formatNumber(model.rmse, 2)}</b></div>
              </div>
              <p>Wh per 10-minute interval</p>
            </Reveal>
          ))}
        </div>
        <Reveal className="final-score">
          <div className="score-label">
            <p className="eyebrow light">FINAL UNSEEN TEST</p>
            <h3>Baseline <span>vs.</span> {siteData.selectedModel.name}</h3>
            <p>{siteData.periods.test.rows.toLocaleString()} later observations · identical test period</p>
          </div>
          <div className="score-column muted">
            <span>Baseline RMSE</span>
            <strong>{formatNumber(siteData.finalBaseline.rmse_wh, 2)}</strong>
            <small>MAE {formatNumber(siteData.finalBaseline.mae_wh, 2)} Wh</small>
          </div>
          <div className="score-column bright">
            <span>AI RMSE</span>
            <strong>{formatNumber(siteData.finalModel.rmse_wh, 2)}</strong>
            <small>MAE {formatNumber(siteData.finalModel.mae_wh, 2)} Wh</small>
          </div>
          <div className="improvement-badge">
            <span>RMSE improvement</span>
            <strong><CountUp value={siteData.improvement.rmse_percent} digits={1} suffix="%" /></strong>
            <small>MAE improved {formatNumber(siteData.improvement.mae_percent, 1)}%</small>
          </div>
        </Reveal>
      </section>

      <section className="chart-section section-shell">
        <SectionHeading
          eyebrow="05 / ACTUAL VS EXPECTED"
          title="Two lines. One moment worth noticing."
          description={`A real final-test slice: ${siteData.chart.title}. Red points are flagged observations.`}
        />
        <Reveal className="chart-card">
          <div className="chart-legend">
            <span><i className="actual-key" />Actual</span>
            <span><i className="expected-key" />AI expected</span>
            <span><i className="anomaly-key" />Anomaly</span>
          </div>
          <div className="chart-wrap" role="img" aria-label="Line chart comparing actual and AI expected consumption">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartRows} margin={{ top: 20, right: 12, left: -14, bottom: 4 }}>
                <CartesianGrid stroke="rgba(16,20,15,.09)" vertical={false} />
                <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={35} tick={{ fill: "#667064", fontSize: 12 }} />
                <YAxis tickLine={false} axisLine={false} tick={{ fill: "#667064", fontSize: 12 }} unit=" Wh" width={72} />
                <Tooltip
                  contentStyle={{ borderRadius: 14, border: "1px solid #dbe1d5", boxShadow: "0 16px 40px rgba(16,20,15,.12)" }}
                  formatter={(value, name) => [`${formatNumber(Number(value), 1)} Wh`, name === "actual" ? "Actual" : name === "expected" ? "AI expected" : "Anomaly"]}
                />
                <Line type="monotone" dataKey="expected" stroke="#669f50" strokeWidth={3} dot={false} activeDot={{ r: 4 }} />
                <Line type="monotone" dataKey="actual" stroke="#10140f" strokeWidth={2.5} dot={false} activeDot={{ r: 4 }} />
                <Scatter dataKey="anomalyActual" fill="#e45439" shape="circle" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="chart-note">Consumption is measured in Wh per 10-minute interval.</p>
        </Reveal>
      </section>

      <section className="explorer section-shell" id="explore">
        <SectionHeading
          eyebrow="06 / ANOMALY EXPLORER"
          title="Browse the periods the model could not explain."
          description={`${siteData.anomalies.length} final-test observations exceeded the frozen validation-derived threshold.`}
        />
        <div className="explorer-layout">
          <div className="explorer-controls">
            <div className="explorer-count">
              <strong>{String(anomalyIndex + 1).padStart(2, "0")}</strong>
              <span>/ {String(siteData.anomalies.length).padStart(2, "0")}</span>
            </div>
            <div className="arrow-buttons">
              <button type="button" onClick={() => moveAnomaly(-1)} aria-label="Previous anomaly">←</button>
              <button type="button" onClick={() => moveAnomaly(1)} aria-label="Next anomaly">→</button>
            </div>
            <p>Sorted by largest positive residual.</p>
          </div>
          <AnimatePresence mode="wait"><AnomalyCard anomaly={anomaly} /></AnimatePresence>
        </div>
      </section>

      <section className="context section-shell">
        <SectionHeading
          eyebrow="07 / CONTEXT MATTERS"
          title="High is not always unusual. Lower is not always normal."
          description="A fixed threshold cannot tell these two real observations apart intelligently."
        />
        <div className="context-grid">
          <Reveal className="context-card normal">
            <div className="context-card-label"><span />High, but expected</div>
            <h3>{formatNumber(siteData.context.highNormal.actual, 0)} Wh</h3>
            <p>AI expected {formatNumber(siteData.context.highNormal.expected, 1)} Wh</p>
            <div className="context-track"><span style={{ width: `${Math.min(100, (siteData.context.highNormal.actual / 500) * 100)}%` }} /></div>
            <strong>{formatNumber(siteData.context.highNormal.residual, 1)} Wh residual</strong>
            <small>{formatTimestamp(siteData.context.highNormal.timestamp)}</small>
          </Reveal>
          <Reveal className="context-card unusual" delay={0.08}>
            <div className="context-card-label"><span />Lower, but unexpected</div>
            <h3>{formatNumber(siteData.context.lowerUnusual.actual, 0)} Wh</h3>
            <p>AI expected {formatNumber(siteData.context.lowerUnusual.expected, 1)} Wh</p>
            <div className="context-track"><span style={{ width: `${Math.min(100, (siteData.context.lowerUnusual.actual / 500) * 100)}%` }} /></div>
            <strong>+{formatNumber(siteData.context.lowerUnusual.residual, 1)} Wh residual</strong>
            <small>{formatTimestamp(siteData.context.lowerUnusual.timestamp)}</small>
          </Reveal>
        </div>
      </section>

      <section className="failure section-shell">
        <SectionHeading
          eyebrow="08 / FAILURE LAB"
          title="AI gets things wrong."
          description="The three largest final-test errors. Large misses are documented, not hidden."
        />
        <div className="failure-list">
          {siteData.failures.slice(0, 3).map((failure, index) => (
            <Reveal className="failure-row" key={failure.timestamp} delay={index * 0.06}>
              <span className="failure-rank">0{index + 1}</span>
              <div><span>When</span><strong>{formatTimestamp(failure.timestamp)}</strong></div>
              <div><span>Predicted</span><strong>{formatNumber(failure.expected, 1)} Wh</strong></div>
              <div><span>Actual</span><strong>{formatNumber(failure.actual, 1)} Wh</strong></div>
              <div className="error-value"><span>Absolute error</span><strong>{formatNumber(failure.absoluteError, 1)} Wh</strong></div>
            </Reveal>
          ))}
        </div>
        <Reveal className="failure-note">
          <h3>What the model cannot observe</h3>
          <div>{["Occupancy", "Individual appliance activity", "Maintenance events", "Unusual behavior"].map((item) => <span key={item}>{item}</span>)}</div>
          <p>These are missing-information examples—not proven causes of any error.</p>
        </Reveal>
      </section>

      <section className="limitations section-shell">
        <SectionHeading eyebrow="09 / LIMITATIONS" title="Useful signals, honest boundaries." />
        <div className="limitations-grid">
          {[
            ["01", "An anomaly is not proof of waste."],
            ["02", "The dataset reflects one specific residential energy environment."],
            ["03", "The model cannot identify individual appliances."],
            ["04", "Unseen behavior and changing conditions can reduce accuracy."],
            ["05", "Production deployment would require monitoring and retraining."],
          ].map(([number, text]) => (
            <Reveal className="limitation" key={number}>
              <span>{number}</span><p>{text}</p>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="closing section-shell">
        <Reveal className="closing-card">
          <p className="eyebrow light">WATTWISE</p>
          <h2>See energy differently.</h2>
          <p>AI-powered energy intelligence for buildings.</p>
          <a className="primary-button" href="#top">Back to the signal ↑</a>
        </Reveal>
      </section>

      <footer>
        <div className="brand"><span className="brand-mark">W</span>WATTWISE</div>
        <p>Built from UCI Appliances Energy Prediction data · Wh per 10-minute interval</p>
        <p>Veritas AI International High School Hackathon 2026 · AI + Climate</p>
      </footer>
    </main>
  );
}
