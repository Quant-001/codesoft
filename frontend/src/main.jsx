import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  BadgeCheck,
  Building2,
  ChevronRight,
  CircleDollarSign,
  CreditCard,
  Loader2,
  MapPin,
  Radar,
  ShieldCheck,
  SlidersHorizontal,
} from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8765";

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

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const riskPercent = result ? Math.round(result.fraud_probability * 100) : 0;
  const resultClass = result?.prediction === 1 ? "danger" : "safe";
  const statusLabel = result ? result.risk_level : "Ready";

  const distance = useMemo(
    () => calculateDistance(form.lat, form.long, form.merch_lat, form.merch_long),
    [form.lat, form.long, form.merch_lat, form.merch_long],
  );

  function updateField(event) {
    const { name, value, type } = event.target;
    setForm((current) => ({
      ...current,
      [name]: type === "number" ? Number(value) : value,
    }));
  }

  async function submitTransaction(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed");
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
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
          <h1>Credit Card Fraud Detection</h1>
          <p className="summary">
            Screen transactions with a trained model, review risk signals, and
            make a clear approval decision from one focused workspace.
          </p>
        </div>
        <div className="hero-actions" aria-label="System status">
          <span>
            <Radar size={16} />
            {statusLabel}
          </span>
          <span>
            <CreditCard size={16} />
            Live scoring
          </span>
        </div>
      </section>

      <section className="stats-grid" aria-label="Transaction snapshot">
        <Metric icon={CircleDollarSign} label="Amount" value={`$${form.amt.toFixed(2)}`} />
        <Metric icon={Building2} label="Merchant" value={form.merchant.replace("fraud_", "")} />
        <Metric icon={MapPin} label="Distance" value={`${Math.round(distance)} mi`} />
        <Metric icon={SlidersHorizontal} label="Category" value={form.category} />
      </section>

      <section className="workspace">
        <form className="panel form-grid" onSubmit={submitTransaction}>
          <div className="panel-title">
            <span>Transaction Intake</span>
            <CreditCard size={20} />
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
                <span>{result.model_source}</span>
              </div>
              <p className="result-copy">{result.recommendation}</p>

              <div className="risk-list">
                <span className="risk-list-title">Risk Signals</span>
                {result.risk_factors.map((factor) => (
                  <p key={factor}>
                    <ChevronRight size={16} />
                    {factor}
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
