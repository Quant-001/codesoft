import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  BadgeCheck,
  BarChart3,
  BookOpenText,
  Building2,
  CheckCircle2,
  ChevronRight,
  CircleDollarSign,
  Clock3,
  CreditCard,
  Database,
  Info,
  Loader2,
  MapPin,
  Radar,
  ShieldCheck,
  SlidersHorizontal,
  XCircle,
} from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "/api";

const categories = [
  "shopping_net",
  "misc_net",
  "grocery_net",
  "gas_transport",
  "food_dining",
  "health_fitness",
  "entertainment",
  "travel",
];

const states = ["NC", "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "AZ"];

const initialForm = {
  amt: 129.85,
  category: "shopping_net",
  merchant: "fraud_Kilback LLC",
  gender: "F",
  city: "Moravian Falls",
  state: "NC",
  job: "Psychologist",
  lat: 36.0788,
  long: -81.1781,
  merch_lat: 36.0112,
  merch_long: -82.0483,
  city_pop: 3495,
  unix_time: 1325376018,
};

const transactionPresets = [
  {
    label: "Normal",
    icon: BadgeCheck,
    values: {
      amt: 42.3,
      category: "food_dining",
      merchant: "fraud_Local Cafe",
      merch_lat: 36.0812,
      merch_long: -81.1642,
      city_pop: 3495,
    },
  },
  {
    label: "Online Risk",
    icon: AlertTriangle,
    values: {
      amt: 875.49,
      category: "shopping_net",
      merchant: "fraud_Kilback LLC",
      merch_lat: 39.742,
      merch_long: -104.9915,
      city_pop: 3495,
    },
  },
  {
    label: "Travel",
    icon: MapPin,
    values: {
      amt: 520.2,
      category: "travel",
      merchant: "fraud_Rutherford Group",
      merch_lat: 34.0522,
      merch_long: -118.2437,
      city_pop: 3495,
    },
  },
];

const glossaryTerms = [
  {
    term: "Fraud probability",
    meaning: "The model's estimated chance that this transaction is fraud.",
    example: "72% means the model sees strong fraud-like patterns.",
  },
  {
    term: "Risk level",
    meaning: "A readable label created from the fraud probability.",
    example: "Low, Moderate, Elevated, or Critical.",
  },
  {
    term: "Decision threshold",
    meaning: "The probability limit used to mark a transaction as fraud.",
    example: "With a 55% threshold, 56% becomes Fraudulent.",
  },
  {
    term: "Recommendation",
    meaning: "The suggested action for the analyst based on risk.",
    example: "Approve, monitor, manual review, or block/hold.",
  },
  {
    term: "Risk signals",
    meaning: "Human-readable reasons that explain why risk increased.",
    example: "High amount or merchant far from customer location.",
  },
  {
    term: "Model version",
    meaning: "The name of the trained model currently used by the app.",
    example: "Random Forest v1.0.",
  },
  {
    term: "Accuracy",
    meaning: "How often the model is correct overall.",
    example: "Can look high when fraud is rare, so it is not enough alone.",
  },
  {
    term: "Precision",
    meaning: "Of transactions predicted as fraud, how many were actually fraud.",
    example: "High precision means fewer false fraud alerts.",
  },
  {
    term: "Recall",
    meaning: "Of real fraud transactions, how many the model caught.",
    example: "High recall means fewer missed fraud cases.",
  },
  {
    term: "F1 score",
    meaning: "A balance between precision and recall.",
    example: "Useful when fraud and legitimate transactions are imbalanced.",
  },
  {
    term: "ROC-AUC",
    meaning: "Measures how well the model separates fraud from legitimate transactions.",
    example: "Closer to 1.0 is better.",
  },
  {
    term: "Confusion matrix",
    meaning: "A table showing correct and incorrect predictions.",
    example: "It shows true legit, false fraud, missed fraud, and caught fraud.",
  },
];

const fallbackMetrics = {
  model_version: "Random Forest v1.0",
  model_source: "Demo rules engine",
  threshold: 0.55,
  trained_at: null,
  dataset: "Kaggle fraudTrain.csv",
  sample_rows: null,
  accuracy: null,
  precision: null,
  recall: null,
  f1_score: null,
  roc_auc: null,
  confusion_matrix: [
    [0, 0],
    [0, 0],
  ],
  feature_count: 13,
  positive_rate: null,
};

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(fallbackMetrics);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [decisionLoading, setDecisionLoading] = useState("");
  const [activeView, setActiveView] = useState("score");

  const riskPercent = result ? Math.round(result.fraud_probability * 100) : 0;
  const resultClass = result?.prediction === 1 ? "danger" : "safe";
  const statusLabel = result ? result.risk_level : "Ready";
  const pendingCount = history.filter((item) => item.analyst_decision === "Pending").length;
  const fraudCount = history.filter((item) => item.prediction === 1).length;

  const distance = useMemo(
    () => calculateDistance(form.lat, form.long, form.merch_lat, form.merch_long),
    [form.lat, form.long, form.merch_lat, form.merch_long],
  );

  useEffect(() => {
    refreshOperationalData();
  }, []);

  async function refreshOperationalData() {
    const [transactionData, metricsData] = await Promise.all([
      fetchJson("/transactions?limit=12").catch(() => []),
      fetchJson("/metrics").catch(() => fallbackMetrics),
    ]);
    setHistory(transactionData);
    setMetrics(metricsData);
  }

  function updateField(event) {
    const { name, value, type } = event.target;
    setForm((current) => ({
      ...current,
      [name]: type === "number" ? Number(value) : value,
    }));
  }

  function applyPreset(values) {
    setForm((current) => ({
      ...current,
      ...values,
    }));
  }

  async function submitTransaction(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await fetchJson("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const scoredResult = {
        ...data,
        transaction_id: data.transaction_id ?? Date.now(),
      };
      setResult(scoredResult);
      if (data.transaction_id) {
        await refreshOperationalData();
      } else {
        setHistory((current) => [buildLocalRecord(scoredResult, form), ...current].slice(0, 12));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function submitDecision(decision) {
    if (!result?.transaction_id) {
      setError("Score a transaction before saving an analyst decision.");
      return;
    }

    setDecisionLoading(decision);
    setError("");
    try {
      await fetchJson(`/decisions/${result.transaction_id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision }),
      });
      await refreshOperationalData();
    } catch (err) {
      setHistory((current) =>
        current.map((item) =>
          item.id === result.transaction_id
            ? { ...item, analyst_decision: decision, decided_at: new Date().toISOString() }
            : item,
        ),
      );
    } finally {
      setDecisionLoading("");
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">
            <ShieldCheck size={16} />
            Fraud operations console
          </span>
          <h1>FraudShield Review Desk</h1>
          <p className="summary">
            Score transactions, explain model risk, capture analyst decisions, and
            monitor model quality from one operational workspace.
          </p>
        </div>
        <div className="hero-actions" aria-label="System status">
          <span>
            <Radar size={16} />
            {statusLabel}
          </span>
          <span>
            <Database size={16} />
            {metrics.model_version}
          </span>
          <span>
            <Clock3 size={16} />
            {pendingCount} pending
          </span>
        </div>
      </section>

      <section className="stats-grid" aria-label="Transaction snapshot">
        <Metric icon={CircleDollarSign} label="Amount" value={`$${form.amt.toFixed(2)}`} />
        <Metric icon={Building2} label="Merchant" value={form.merchant.replace("fraud_", "")} />
        <Metric icon={MapPin} label="Distance" value={`${Math.round(distance)} mi`} />
        <Metric icon={AlertTriangle} label="Flagged" value={`${fraudCount} of ${history.length || 0}`} />
      </section>

      <nav className="tabs" aria-label="Dashboard views">
        <button className={activeView === "score" ? "active" : ""} onClick={() => setActiveView("score")}>
          <Radar size={17} />
          Score
        </button>
        <button className={activeView === "history" ? "active" : ""} onClick={() => setActiveView("history")}>
          <Clock3 size={17} />
          History
        </button>
        <button className={activeView === "metrics" ? "active" : ""} onClick={() => setActiveView("metrics")}>
          <BarChart3 size={17} />
          Metrics
        </button>
        <button className={activeView === "terms" ? "active" : ""} onClick={() => setActiveView("terms")}>
          <BookOpenText size={17} />
          Terms
        </button>
      </nav>

      {activeView === "score" && (
        <section className="workspace">
          <form className="panel form-grid" onSubmit={submitTransaction}>
            <div className="panel-title">
              <span>Transaction Intake</span>
              <CreditCard size={20} />
            </div>

            <div className="preset-bar" aria-label="Transaction examples">
              {transactionPresets.map((preset) => {
                const Icon = preset.icon;
                return (
                  <button
                    key={preset.label}
                    type="button"
                    onClick={() => applyPreset(preset.values)}
                    title={`Load ${preset.label} example`}
                  >
                    <Icon size={16} />
                    {preset.label}
                  </button>
                );
              })}
            </div>

            <Field label="Amount" name="amt" type="number" step="0.01" value={form.amt} onChange={updateField} />
            <Select label="Category" name="category" value={form.category} options={categories} onChange={updateField} />
            <Field label="Merchant" name="merchant" value={form.merchant} onChange={updateField} />
            <Select label="Gender" name="gender" value={form.gender} options={["F", "M"]} onChange={updateField} />
            <Field label="City" name="city" value={form.city} onChange={updateField} />
            <Select label="State" name="state" value={form.state} options={states} onChange={updateField} />
            <Field label="Job" name="job" value={form.job} onChange={updateField} />
            <Field label="Customer Latitude" name="lat" type="number" step="0.0001" value={form.lat} onChange={updateField} />
            <Field label="Customer Longitude" name="long" type="number" step="0.0001" value={form.long} onChange={updateField} />
            <Field label="Merchant Latitude" name="merch_lat" type="number" step="0.0001" value={form.merch_lat} onChange={updateField} />
            <Field label="Merchant Longitude" name="merch_long" type="number" step="0.0001" value={form.merch_long} onChange={updateField} />
            <Field label="City Population" name="city_pop" type="number" value={form.city_pop} onChange={updateField} />
            <Field label="Unix Time" name="unix_time" type="number" value={form.unix_time} onChange={updateField} />

            <button className="submit-button" type="submit" disabled={loading}>
              {loading ? <Loader2 className="spin" size={18} /> : <Radar size={18} />}
              Score Transaction
            </button>
          </form>

          <aside className={`panel result-panel ${result ? resultClass : ""}`}>
            <div className="panel-title">
              <span>Analyst Decision</span>
              {result?.prediction === 1 ? (
                <AlertTriangle className="danger" size={24} />
              ) : (
                <BadgeCheck className="safe" size={24} />
              )}
            </div>

            {result ? (
              <>
                <div className="decision">
                  <span className="risk-label">{result.risk_level} risk</span>
                  <strong className={result.prediction === 1 ? "danger-text" : "safe-text"}>
                    {result.label}
                  </strong>
                </div>
                <div className="meter" aria-label={`Fraud probability ${riskPercent}%`}>
                  <span style={{ width: `${riskPercent}%` }} />
                </div>
                <div className="score-row">
                  <span>{riskPercent}% fraud probability</span>
                  <span>{Math.round(result.threshold * 100)}% threshold</span>
                </div>
                <p className="result-copy">{result.recommendation}</p>

                <div className="decision-actions">
                  <DecisionButton icon={CheckCircle2} label="Approve" loading={decisionLoading} onClick={submitDecision} />
                  <DecisionButton icon={AlertTriangle} label="Manual Review" loading={decisionLoading} onClick={submitDecision} />
                  <DecisionButton icon={XCircle} label="Reject" loading={decisionLoading} onClick={submitDecision} />
                </div>

                <div className="risk-list">
                  <span className="risk-list-title">Explainable Risk Signals</span>
                  {result.explanations.map((factor) => (
                    <p key={factor.signal}>
                      <ChevronRight size={16} />
                      <span>{factor.signal}</span>
                      <b>{factor.impact} +{Math.round(factor.contribution * 100)}%</b>
                    </p>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-state">
                <ShieldCheck size={36} />
                <p>Submit a transaction to see fraud probability, risk level, and review signals.</p>
              </div>
            )}

            {error && <p className="error">{error}</p>}
          </aside>
        </section>
      )}

      {activeView === "history" && (
        <HistoryView history={history} onRefresh={refreshOperationalData} />
      )}

      {activeView === "metrics" && <MetricsView metrics={metrics} />}

      {activeView === "terms" && <TermsView terms={glossaryTerms} />}
    </main>
  );
}

function Metric({ icon: Icon, label, value }) {
  return (
    <div className="metric">
      <Icon size={20} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Field({ label, ...props }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input required {...props} />
    </label>
  );
}

function Select({ label, options, ...props }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select required {...props}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function DecisionButton({ icon: Icon, label, loading, onClick }) {
  return (
    <button type="button" onClick={() => onClick(label)} disabled={Boolean(loading)}>
      {loading === label ? <Loader2 className="spin" size={16} /> : <Icon size={16} />}
      {label}
    </button>
  );
}

function HistoryView({ history, onRefresh }) {
  return (
    <section className="panel table-panel">
      <div className="panel-title">
        <span>Transaction History</span>
        <button className="ghost-button" type="button" onClick={onRefresh}>
          <Clock3 size={16} />
          Refresh
        </button>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Merchant</th>
              <th>Amount</th>
              <th>Risk</th>
              <th>Probability</th>
              <th>Decision</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {history.map((item) => (
              <tr key={item.id}>
                <td>#{item.id}</td>
                <td>
                  <strong>{item.merchant.replace("fraud_", "")}</strong>
                  <span>{item.category}</span>
                </td>
                <td>${Number(item.amount).toFixed(2)}</td>
                <td>
                  <span className={`pill ${item.prediction === 1 ? "danger-pill" : "safe-pill"}`}>
                    {item.risk_level}
                  </span>
                </td>
                <td>{Math.round(item.fraud_probability * 100)}%</td>
                <td>{item.analyst_decision}</td>
                <td>{new Date(item.created_at).toLocaleString()}</td>
              </tr>
            ))}
            {history.length === 0 && (
              <tr>
                <td colSpan="7" className="empty-table">
                  No saved transactions yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MetricsView({ metrics }) {
  const confusion = metrics.confusion_matrix || [
    [0, 0],
    [0, 0],
  ];

  return (
    <section className="metrics-layout">
      <div className="panel metrics-card">
        <div className="panel-title">
          <span>Model Quality</span>
          <BarChart3 size={20} />
        </div>
        <div className="quality-grid">
          <Quality label="Accuracy" value={metrics.accuracy} />
          <Quality label="Precision" value={metrics.precision} />
          <Quality label="Recall" value={metrics.recall} />
          <Quality label="F1 Score" value={metrics.f1_score} />
          <Quality label="ROC-AUC" value={metrics.roc_auc} />
          <Quality label="Fraud Rate" value={metrics.positive_rate} />
        </div>
      </div>

      <div className="panel metrics-card">
        <div className="panel-title">
          <span>Model Versioning</span>
          <Database size={20} />
        </div>
        <dl className="metadata">
          <div>
            <dt>Model</dt>
            <dd>{metrics.model_version}</dd>
          </div>
          <div>
            <dt>Source</dt>
            <dd>{metrics.model_source}</dd>
          </div>
          <div>
            <dt>Decision threshold</dt>
            <dd>{Math.round(metrics.threshold * 100)}%</dd>
          </div>
          <div>
            <dt>Dataset</dt>
            <dd>{metrics.dataset}</dd>
          </div>
          <div>
            <dt>Training rows</dt>
            <dd>{metrics.sample_rows ? metrics.sample_rows.toLocaleString() : "Not trained"}</dd>
          </div>
          <div>
            <dt>Features</dt>
            <dd>{metrics.feature_count}</dd>
          </div>
        </dl>
      </div>

      <div className="panel metrics-card confusion-card">
        <div className="panel-title">
          <span>Confusion Matrix</span>
          <ShieldCheck size={20} />
        </div>
        <div className="confusion-grid">
          <span />
          <b>Pred Legit</b>
          <b>Pred Fraud</b>
          <b>Actual Legit</b>
          <strong>{confusion[0]?.[0] ?? 0}</strong>
          <strong>{confusion[0]?.[1] ?? 0}</strong>
          <b>Actual Fraud</b>
          <strong>{confusion[1]?.[0] ?? 0}</strong>
          <strong>{confusion[1]?.[1] ?? 0}</strong>
        </div>
      </div>
    </section>
  );
}

function TermsView({ terms }) {
  return (
    <section className="terms-layout">
      <div className="panel terms-intro">
        <div className="panel-title">
          <span>UI Terms Explained</span>
          <Info size={20} />
        </div>
        <p>
          These are the words used in the dashboard. They help you understand what
          the fraud model predicted, why it predicted it, and how strong the result is.
        </p>
      </div>

      <div className="terms-grid">
        {terms.map((item) => (
          <article className="term-card" key={item.term}>
            <h2>{item.term}</h2>
            <p>{item.meaning}</p>
            <span>{item.example}</span>
          </article>
        ))}
      </div>
    </section>
  );
}

function Quality({ label, value }) {
  const display = value === null || value === undefined ? "N/A" : `${Math.round(value * 100)}%`;
  return (
    <div className="quality">
      <span>{label}</span>
      <strong>{display}</strong>
    </div>
  );
}

async function fetchJson(path, options) {
  const response = await fetch(`${API_URL}${path}`, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function buildLocalRecord(result, transaction) {
  return {
    id: result.transaction_id,
    created_at: new Date().toISOString(),
    amount: transaction.amt,
    merchant: transaction.merchant,
    category: transaction.category,
    city: transaction.city,
    state: transaction.state,
    prediction: result.prediction,
    label: result.label,
    fraud_probability: result.fraud_probability,
    risk_level: result.risk_level,
    model_source: result.model_source,
    model_version: result.model_version,
    threshold: result.threshold,
    analyst_decision: "Pending",
    decision_note: null,
    decided_at: null,
    request_payload: transaction,
    risk_factors: result.risk_factors,
    explanations: result.explanations,
  };
}

function calculateDistance(lat1, lon1, lat2, lon2) {
  const toRadians = (value) => (Number(value) * Math.PI) / 180;
  const radiusMiles = 3958.8;
  const dlat = toRadians(lat2 - lat1);
  const dlon = toRadians(lon2 - lon1);
  const a =
    Math.sin(dlat / 2) ** 2 +
    Math.cos(toRadians(lat1)) *
      Math.cos(toRadians(lat2)) *
      Math.sin(dlon / 2) ** 2;

  return radiusMiles * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

createRoot(document.getElementById("root")).render(<App />);
