import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import "./style.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const PRIORITY_COLORS = {
  CRITICAL: "#ef4444",
  HIGH: "#f97316",
  MEDIUM: "#eab308",
  LOW: "#22c55e",
};

function playAlertChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.35);
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.35);
  } catch {
    // Ignore audio autoplay restrictions
  }
}

function getHealthClass(score) {
  if (score >= 85) return "health-excellent";
  if (score >= 65) return "health-good";
  if (score >= 40) return "health-warning";
  return "health-critical";
}

function getMaintTagClass(category) {
  const cat = (category || "").toUpperCase();
  if (cat === "URGENT") return "maint-urgent";
  if (cat === "PREVENTIVE") return "maint-preventive";
  return "maint-routine";
}

function App() {
  const [activePage, setActivePage] = useState("Overview");
  const [selectedMachine, setSelectedMachine] = useState(null);
  const [showReportModal, setShowReportModal] = useState(false);
  const [reportData, setReportData] = useState(null);
  const [reportError, setReportError] = useState("");

  const [overview, setOverview] = useState(null);
  const [machines, setMachines] = useState([]);
  const [modelHealth, setModelHealth] = useState(null);
  const [components, setComponents] = useState([]);
  const [failureMatches, setFailureMatches] = useState([]);
  const [historicalTrends, setHistoricalTrends] = useState(null);
  const [productionLines, setProductionLines] = useState([]);
  const [scenarios, setScenarios] = useState([]);
  const [activeScenarioId, setActiveScenarioId] = useState("continuous_replay");
  const [downtimeTrial, setDowntimeTrial] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [simulation, setSimulation] = useState(null);
  const [simulationPlaying, setSimulationPlaying] = useState(false);
  const [simulationCapacity, setSimulationCapacity] = useState(5);
  const [audioAlertsEnabled, setAudioAlertsEnabled] = useState(false);
  const [simulationError, setSimulationError] = useState("");

  const [search, setSearch] = useState("");
  const [priorityFilter, setPriorityFilter] = useState("ALL");
  const [lineFilter, setLineFilter] = useState("ALL");

  const navigation = [
    { name: "Overview", icon: "◫" },
    { name: "Factory Floor", icon: "🏭" },
    { name: "Machines", icon: "▦" },
    { name: "Predictive Alerts", icon: "⚠" },
    { name: "Maintenance", icon: "⚙" },
    { name: "Operational Trial", icon: "🔬" },
    { name: "Analytics", icon: "▥" },
  ];

  // Load initial data
  useEffect(() => {
    async function loadDashboard() {
      try {
        setLoading(true);
        setError("");

        const [
          overviewRes,
          machinesRes,
          modelRes,
          componentsRes,
          failureMatchesRes,
          trendsRes,
          linesRes,
          scenariosRes,
          trialRes,
          replayStateRes,
        ] = await Promise.all([
          fetch(`${API_BASE}/api/overview`),
          fetch(`${API_BASE}/api/priorities`),
          fetch(`${API_BASE}/api/model/health`),
          fetch(`${API_BASE}/api/components`),
          fetch(`${API_BASE}/api/failure_matches`),
          fetch(`${API_BASE}/api/historical_trends`).catch(() => ({ ok: false })),
          fetch(`${API_BASE}/api/production_lines`).catch(() => ({ ok: false })),
          fetch(`${API_BASE}/api/simulation/scenarios`).catch(() => ({ ok: false })),
          fetch(`${API_BASE}/api/trial/downtime_reduction`).catch(() => ({ ok: false })),
          fetch(`${API_BASE}/api/simulation/state?machines_per_day=${simulationCapacity}`).catch(() => ({ ok: false })),
        ]);

        if (!overviewRes.ok || !machinesRes.ok || !modelRes.ok) {
          throw new Error("Could not load core dashboard data from FastAPI.");
        }

        const overviewData = await overviewRes.json();
        const machinesData = await machinesRes.json();
        const modelData = await modelRes.json();
        const componentsData = await componentsRes.json();
        const failureMatchesData = await failureMatchesRes.json();

        setOverview(overviewData);
        setMachines(machinesData.items || []);
        setModelHealth(modelData);
        if (replayStateRes.ok) {
          const replayStateData = await replayStateRes.json();
          if (replayStateData.snapshot) setSimulation(replayStateData.snapshot);
        }
        setComponents(componentsData.items || []);
        setFailureMatches(failureMatchesData.items || []);

        if (trendsRes.ok) {
          const trendsData = await trendsRes.json();
          setHistoricalTrends(trendsData);
        }
        if (linesRes.ok) {
          const linesData = await linesRes.json();
          setProductionLines(linesData.lines || []);
        }
        if (scenariosRes.ok) {
          const scData = await scenariosRes.json();
          setScenarios(scData.scenarios || []);
        }
        if (trialRes.ok) {
          const trialData = await trialRes.json();
          setDowntimeTrial(trialData);
        }
      } catch (err) {
        console.error("Dashboard loading error:", err);
        setError("Unable to connect to the backend API. Make sure FastAPI is running on http://127.0.0.1:8000.");
      } finally {
        setLoading(false);
      }
    }

    loadDashboard();
  }, []);

  // Advance simulation step
  const stepSimulation = useCallback(async () => {
    try {
      setSimulationError("");
      const response = await fetch(`${API_BASE}/api/simulation/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ machines_per_day: simulationCapacity }),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(
          typeof result.detail === "string" ? result.detail : "Could not advance the historical replay."
        );
      }
      setSimulation(result);

      if (audioAlertsEnabled && result.overview?.alerts > 0) {
        playAlertChime();
      }

      if (result.progress.complete) setSimulationPlaying(false);
    } catch (requestError) {
      setSimulationPlaying(false);
      setSimulationError(requestError.message);
    }
  }, [simulationCapacity, audioAlertsEnabled]);

  // Simulation auto-timer
  useEffect(() => {
    if (!simulationPlaying) return undefined;
    const timer = window.setInterval(stepSimulation, 1300);
    return () => window.clearInterval(timer);
  }, [simulationPlaying, stepSimulation]);

  // Reset simulation
  async function resetSimulation() {
    setSimulationPlaying(false);
    setSimulationError("");
    try {
      const response = await fetch(`${API_BASE}/api/simulation/reset`, { method: "POST" });
      if (!response.ok) throw new Error("Could not reset the replay.");
      setSimulation(null);
      setActiveScenarioId("continuous_replay");
    } catch (requestError) {
      setSimulationError(requestError.message);
    }
  }

  // Load a preset factory scenario
  async function handleLoadScenario(scenarioId) {
    try {
      setSimulationPlaying(false);
      setSimulationError("");
      setActiveScenarioId(scenarioId);
      const res = await fetch(`${API_BASE}/api/simulation/scenario/${scenarioId}?machines_per_day=${simulationCapacity}`, {
        method: "POST",
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.detail || "Failed to load scenario.");
      setSimulation(result);
      if (audioAlertsEnabled && result.overview?.alerts > 0) {
        playAlertChime();
      }
    } catch (err) {
      setSimulationError(err.message);
    }
  }

  // Load dispatch report data
  async function handleOpenReport() {
    try {
      const res = await fetch(`${API_BASE}/api/maintenance/report?machines_per_day=${simulationCapacity}`);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : "Could not generate the dispatch proposal.");
      }
      setReportData(data);
      setReportError("");
    } catch (requestError) {
      setReportData(null);
      setReportError(requestError.message);
    }
    setShowReportModal(true);
  }

  const displayedOverview = simulation?.overview || overview;
  const rawDisplayedMachines = simulation?.machines || machines;

  const alertThreshold = modelHealth?.risk_configuration?.operational_alert_threshold ?? 0.98;

  const displayedMachines = rawDisplayedMachines;

  const activeAlerts = useMemo(() => {
    return displayedMachines.filter((machine) => machine.alert === true);
  }, [displayedMachines]);

  // Search & Filter
  const filteredMachines = useMemo(() => {
    return displayedMachines.filter((machine) => {
      const searchTerm = search.trim().toLowerCase();
      const matchesSearch =
        String(machine.machineID).includes(searchTerm) ||
        machine.model?.toLowerCase().includes(searchTerm) ||
        machine.production_line?.toLowerCase().includes(searchTerm);

      const matchesPriority =
        priorityFilter === "ALL" ||
        machine.priority === priorityFilter ||
        machine.maintenance_category === priorityFilter;

      const matchesLine = lineFilter === "ALL" || machine.production_line === lineFilter;

      return matchesSearch && matchesPriority && matchesLine;
    });
  }, [displayedMachines, search, priorityFilter, lineFilter]);

  // Inspection schedule
  const displayedSchedule = simulation?.schedule || [];

  // Priority chart data
  const priorityData = displayedOverview
    ? [
        { name: "Critical", machines: displayedOverview.critical || 0, color: PRIORITY_COLORS.CRITICAL },
        { name: "High", machines: displayedOverview.high || 0, color: PRIORITY_COLORS.HIGH },
        { name: "Medium", machines: displayedOverview.medium || 0, color: PRIORITY_COLORS.MEDIUM },
        { name: "Low", machines: displayedOverview.low || 0, color: PRIORITY_COLORS.LOW },
      ]
    : [];

  const componentChartData = components.map((item) => ({
    name: `${item.model} - ${item.failure}`,
    failures: item.failure_count,
  }));

  // Analytics derivations
  const averageWarningHours = failureMatches.length
    ? failureMatches.reduce((sum, item) => sum + Number(item.earliest_warning_hours || 0), 0) / failureMatches.length
    : 0;

  const averagePeakProbability = failureMatches.length
    ? failureMatches.reduce((sum, item) => sum + Number(item.peak_probability || 0), 0) / failureMatches.length
    : 0;

  const warningLeadTimeData = [
    { range: "0–6 hrs", count: 0 },
    { range: "7–12 hrs", count: 0 },
    { range: "13–18 hrs", count: 0 },
    { range: "19–24 hrs", count: 0 },
    { range: ">24 hrs", count: 0 },
  ];
  failureMatches.forEach((item) => {
    const hours = Number(item.earliest_warning_hours || 0);
    if (hours <= 6) warningLeadTimeData[0].count += 1;
    else if (hours <= 12) warningLeadTimeData[1].count += 1;
    else if (hours <= 18) warningLeadTimeData[2].count += 1;
    else if (hours <= 24) warningLeadTimeData[3].count += 1;
    else warningLeadTimeData[4].count += 1;
  });

  return (
    <div className="app">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">M</div>
          <div>
            <h2>Manufacture<span>.AI</span></h2>
            <small>PREDICTIVE OPTIMIZATION</small>
          </div>
        </div>

        <div className="nav-label">FACTORY WORKSPACE</div>
        <nav>
          {navigation.map((item) => (
            <button
              key={item.name}
              className={`nav-item ${activePage === item.name ? "active" : ""}`}
              onClick={() => setActivePage(item.name)}
            >
              <span>{item.icon}</span>
              {item.name}
              {item.name === "Predictive Alerts" && activeAlerts.length > 0 && (
                <span className="alert-badge" style={{ marginLeft: "auto" }}>
                  {activeAlerts.length}
                </span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot" />
            <div>
              <strong>{modelHealth?.model_loaded ? "3 Models Active" : "Models Offline"}</strong>
              <small>Binary · RUL · Severity</small>
            </div>
          </div>
          <div className="profile">
            <div className="avatar">PM</div>
            <div>
              <strong>Plant Engineering</strong>
              <small>Production Line 1–5</small>
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main">
        {/* TOPBAR */}
        <header className="topbar">
          <div>
            <span className="breadcrumb">Factory Workspace / </span>
            <strong>{activePage}</strong>
          </div>

          <div className="top-actions">
            <button
              type="button"
              className="alert-btn"
              style={{ color: "#334155", background: "#f1f5f9", borderColor: "#cbd5e1" }}
              onClick={() => setAudioAlertsEnabled(!audioAlertsEnabled)}
              title="Toggle Audible Alert Chime"
            >
              {audioAlertsEnabled ? "🔔 Chime Active" : "🔕 Muted"}
            </button>
            <button type="button" className="view-button" onClick={handleOpenReport}>
              📄 Dispatch Report
            </button>
            <span className="live-indicator">
              <span className="status-dot" />
              {modelHealth?.model_loaded ? "Historical Replay Ready" : "Backend Offline"}
            </span>
          </div>
        </header>

        {/* CONTENT */}
        <section className="content">
          {/* ACTIVE ALERT BANNER MECHANISM */}
          {activeAlerts.length > 0 && (
            <div className="alert-banner" role="alert">
              <div className="alert-banner-content">
                <span className="alert-badge">⚠ ALERT TRIGGERED</span>
                <div>
                  <strong>{activeAlerts.length} machine(s) crossed risk threshold ({Math.round(alertThreshold * 100)}%)!</strong>
                  <div style={{ fontSize: "12px", opacity: 0.9 }}>
                    Model-based alerts from the current replay snapshot; qualified inspection is required before action.
                  </div>
                </div>
              </div>
              <div className="alert-banner-actions">
                <button
                  type="button"
                  className="alert-btn"
                  onClick={() => {
                    setActivePage("Predictive Alerts");
                    setPriorityFilter("ALL");
                  }}
                >
                  View Alerting Queue →
                </button>
              </div>
            </div>
          )}

          {/* PAGE TITLE */}
          <div className="page-heading">
            <div>
              <p className="eyebrow">AI-DRIVEN PREDICTIVE MAINTENANCE</p>
              <h1>{activePage}</h1>
              <p className="subtitle">
                Predictions and priority suggestions are generated for historical-data replay; this is not a live factory feed.
              </p>
            </div>
            <div className="date-label">
              MULTI-MODEL SUITE: 24H HORIZON &bull; RUL REGRESSION
            </div>
          </div>

          {/* SIMULATED FACTORY SCENARIO TOOLBAR */}
          <section className="simulation-toolbar" aria-label="Simulated factory scenario controller">
            <div className="simulation-copy">
              <strong>Simulated Factory Operations</strong>
              <p>
                {simulation
                  ? `Snapshot ${simulation.timestamp} · Step ${simulation.progress.step.toLocaleString()} of ${simulation.progress.total_steps.toLocaleString()}`
                  : "Historical sensor replay with simulated controls; no live industrial telemetry is connected."}
              </p>
              <small>
                Operational alert threshold: <strong>{(alertThreshold * 100).toFixed(0)}%</strong> &bull;
                Risk bands and alert threshold are configured by the backend.
              </small>

              {/* Scenario Preset Buttons */}
              <div className="scenario-selector-bar">
                <span style={{ fontSize: "12px", fontWeight: 700, color: "#475569" }}>Scenarios:</span>
                {scenarios.map((sc) => (
                  <button
                    key={sc.id}
                    type="button"
                    className={`scenario-btn ${activeScenarioId === sc.id ? "active" : ""}`}
                    onClick={() => handleLoadScenario(sc.id)}
                    title={sc.description}
                  >
                    {sc.name}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
              <label style={{ fontSize: "12px", display: "flex", flexDirection: "column" }}>
                <span>Daily Slots</span>
                <input
                  type="number"
                  min="1"
                  max="25"
                  value={simulationCapacity}
                  onChange={(e) => setSimulationCapacity(Math.max(1, Number(e.target.value) || 1))}
                  style={{ width: "65px" }}
                />
              </label>
            </div>

            {!simulationPlaying ? (
              <button
                type="button"
                className="view-button"
                onClick={() => setSimulationPlaying(true)}
                disabled={simulation?.progress.complete}
              >
                {simulation?.progress.complete ? "Replay Complete" : simulation ? "Resume Replay" : "Start Replay"}
              </button>
            ) : (
              <button type="button" className="view-button" onClick={() => setSimulationPlaying(false)}>
                Pause
              </button>
            )}

            <button type="button" className="view-button" onClick={stepSimulation} disabled={simulationPlaying}>
              Advance 1 Hr
            </button>
            <button type="button" className="view-button" onClick={resetSimulation}>
              Reset
            </button>

            {simulationError && (
              <p className="simulation-error" role="alert" style={{ width: "100%", marginTop: "6px" }}>
                {simulationError}
              </p>
            )}
          </section>

          {/* LOADING & ERROR */}
          {loading && (
            <section className="panel"><p>Loading manufacturing telemetry and model states...</p></section>
          )}
          {error && (
            <section className="panel">
              <p style={{ color: "#ef4444" }}>{error}</p>
              <button className="view-button" onClick={() => window.location.reload()}>Retry connection</button>
            </section>
          )}

          {/* MAIN APPLICATION PAGES */}
          {!loading && !error && displayedOverview && (
            <>
              {/* ======================================================
                  1. OVERVIEW PAGE
                  ====================================================== */}
              {activePage === "Overview" && (
                <>
                  <section className="stats-grid">
                    <StatCard
                      title="Total Equipment"
                      value={displayedOverview.total_machines ?? "—"}
                      note="Machines represented by the checked-in ranking data"
                      icon="▦"
                    />
                    <StatCard
                      title="Factory Health Score"
                      value={displayedOverview.average_health_score == null ? "Not measured" : `${displayedOverview.average_health_score}%`}
                      note={simulation ? "Average project-defined index for this replay snapshot" : "No fleet-wide health score is calculated from saved rankings"}
                      icon="♥"
                      color="blue"
                    />
                    <StatCard
                      title="Urgent Attention"
                      value={simulation ? displayedOverview.critical : "—"}
                      note={simulation ? "Current replay risk-band count" : "Start replay for current model predictions"}
                      icon="⚠"
                      color="red"
                    />
                    <StatCard
                      title="Preventive Risk Band"
                      value={simulation ? displayedOverview.preventive : "—"}
                      note="Current replay HIGH/MEDIUM count; not confirmed work orders"
                      icon="↗"
                      color="orange"
                    />
                    <StatCard
                      title="Estimated Historical Downtime"
                      value={`${overview?.total_historical_downtime_hours ?? "—"} hrs`}
                      note="Component-duration assumption; not actual or avoided downtime"
                      icon="🛡"
                      color="blue"
                    />
                  </section>

                  {/* FACTORY PRODUCTION LINES SUMMARY */}
                  <section className="panel" style={{ marginTop: "20px" }}>
                    <div className="panel-heading">
                      <div>
                        <h3>Saved Machine Priority by Production Line</h3>
                        <p>Grouping of the checked-in machine metadata and priority artifact; line health is not measured.</p>
                      </div>
                      <button className="view-button" onClick={() => setActivePage("Factory Floor")}>
                        View Floor Map →
                      </button>
                    </div>

                    <div className="factory-lines-grid">
                      {productionLines.map((line) => (
                        <div className="factory-line-card" key={line.line_name}>
                          <div className="factory-line-header">
                            <strong>{line.line_name}</strong>
                            <span className="health-pill">
                              {line.health_score == null ? "Health not measured" : `${line.health_score}%`}
                            </span>
                          </div>
                          <div style={{ fontSize: "12px", color: "#64748b" }}>
                            {line.critical_count > 0 ? (
                              <span style={{ color: "#ef4444", fontWeight: 700 }}>
                                ⚠ {line.critical_count} critical machine(s)
                              </span>
                            ) : (
                              <span>No saved CRITICAL priority</span>
                            )}
                          </div>
                          <div className="machine-mini-grid">
                            {line.machines.map((m) => {
                              const p = m.priority;
                              const bg = PRIORITY_COLORS[p] || "#22c55e";
                              return (
                                <div
                                  key={m.machineID}
                                  className="machine-mini-node"
                                  style={{ background: bg }}
                                  title={`Machine ${m.machineID} (${m.model}) - Priority: ${p}`}
                                  onClick={() => {
                                    const full = displayedMachines.find((x) => x.machineID === m.machineID) || m;
                                    setSelectedMachine(full);
                                  }}
                                >
                                  {m.machineID}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>

                  {/* CHARTS: PRIORITY DISTRIBUTION & COMPOSITION */}
                  <section className="charts-grid" style={{ marginTop: "20px" }}>
                    <div className="panel">
                      <div className="panel-heading">
                        <div>
                          <h3>Machine Priority Distribution</h3>
                          <p>{simulation ? "Current replay probability bands" : "Saved historical priority artifact"}</p>
                        </div>
                        <span className="panel-tag">{displayedOverview.total_machines ?? "—"} MACHINES</span>
                      </div>
                      <div className="chart">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={priorityData}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e9edf3" />
                            <XAxis dataKey="name" axisLine={false} tickLine={false} />
                            <YAxis axisLine={false} tickLine={false} />
                            <Tooltip />
                            <Bar dataKey="machines" radius={[6, 6, 0, 0]}>
                              {priorityData.map((entry) => (
                                <Cell key={entry.name} fill={entry.color} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="panel">
                      <div className="panel-heading">
                        <div>
                          <h3>Machine Priority Composition</h3>
                          <p>{simulation ? "Current replay risk bands" : "Breakdown in saved priority data"}</p>
                        </div>
                      </div>
                      <div className="chart pie-chart">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={priorityData}
                              dataKey="machines"
                              nameKey="name"
                              innerRadius={65}
                              outerRadius={95}
                              paddingAngle={4}
                            >
                              {priorityData.map((entry) => (
                                <Cell key={entry.name} fill={entry.color} />
                              ))}
                            </Pie>
                            <Tooltip />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="legend">
                        {priorityData.map((item) => (
                          <span key={item.name}>
                            <i style={{ background: item.color }} />
                            {item.name}: {item.machines}
                          </span>
                        ))}
                      </div>
                    </div>
                  </section>

                  {/* TOP PRIORITY MACHINES TABLE */}
                  <section className="panel machine-panel" style={{ marginTop: "20px" }}>
                    <div className="panel-heading">
                      <div>
                        <h3>{simulation ? "Highest Model-Risk Machines" : "Saved Historical Machine Priorities"}</h3>
                        <p>{simulation ? "Ordered by current failure probability and estimated RUL." : "These ranks are not fresh predictions; start replay to calculate current model risk."}</p>
                      </div>
                      <button className="view-button" onClick={() => setActivePage("Machines")}>
                        View All {displayedMachines.length} Machines →
                      </button>
                    </div>

                    <MachineTable
                      data={displayedMachines.slice(0, 6)}
                      onSelectMachine={setSelectedMachine}
                    />
                  </section>
                </>
              )}

              {/* ======================================================
                  2. FACTORY FLOOR PAGE
                  ====================================================== */}
              {activePage === "Factory Floor" && (
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Machine Layout by Saved Production Line</h3>
                      <p>Historical machine metadata grouped by line. These status colors are saved priorities, not live conditions.</p>
                    </div>
                  </div>

                  <div className="factory-lines-grid" style={{ gridTemplateColumns: "1fr" }}>
                    {productionLines.map((line) => (
                      <div className="factory-line-card" key={line.line_name} style={{ marginBottom: "14px" }}>
                        <div className="factory-line-header">
                          <div>
                            <strong style={{ fontSize: "16px" }}>{line.line_name}</strong>
                            <small style={{ marginLeft: "10px", color: "#64748b" }}>20 Continuous Operating Units</small>
                          </div>
                          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                            <span className="health-pill">
                              {line.health_score == null ? "Health not measured" : `Health: ${line.health_score}%`}
                            </span>
                            <span style={{ fontSize: "12px", color: line.critical_count > 0 ? "#dc2626" : "#16a34a" }}>
                              {line.critical_count > 0 ? `${line.critical_count} saved CRITICAL priority` : "No saved CRITICAL priority"}
                            </span>
                          </div>
                        </div>

                        <div className="machine-mini-grid" style={{ gridTemplateColumns: "repeat(10, 1fr)", gap: "8px" }}>
                          {line.machines.map((m) => {
                            const full = displayedMachines.find((x) => x.machineID === m.machineID) || m;
                            const bg = PRIORITY_COLORS[full.priority] || "#22c55e";
                            return (
                              <div
                                key={m.machineID}
                                className="machine-mini-node"
                                style={{
                                  background: bg,
                                  height: "44px",
                                  fontSize: "12px",
                                  boxShadow: full.alert ? "0 0 10px rgba(239, 68, 68, 0.7)" : "none",
                                }}
                                onClick={() => setSelectedMachine(full)}
                              >
                                #{m.machineID}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* ======================================================
                  3. MACHINES REGISTRY PAGE
                  ====================================================== */}
              {activePage === "Machines" && (
                <section className="panel machine-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Machine Equipment Registry</h3>
                      <p>
                        Showing {filteredMachines.length} of {displayedMachines.length} machines. Historical priorities are available before replay predictions.
                      </p>
                    </div>
                  </div>

                  <div className="machine-filters">
                    <input
                      type="text"
                      placeholder="Search Machine ID, Model, or Production Line..."
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                    />

                    <select value={priorityFilter} onChange={(e) => setPriorityFilter(e.target.value)}>
                      <option value="ALL">All Maintenance Priorities</option>
                      <option value="CRITICAL">Critical Priority</option>
                      <option value="HIGH">High Priority</option>
                      <option value="MEDIUM">Medium Priority</option>
                      <option value="LOW">Low Priority</option>
                      <option value="URGENT">Urgent Maintenance</option>
                      <option value="PREVENTIVE">Preventive Maintenance</option>
                      <option value="ROUTINE">Routine Maintenance</option>
                    </select>

                    <select value={lineFilter} onChange={(e) => setLineFilter(e.target.value)}>
                      <option value="ALL">All Production Lines</option>
                      <option value="Line 1">Line 1 (M1–M20)</option>
                      <option value="Line 2">Line 2 (M21–M40)</option>
                      <option value="Line 3">Line 3 (M41–M60)</option>
                      <option value="Line 4">Line 4 (M61–M80)</option>
                      <option value="Line 5">Line 5 (M81–M100)</option>
                    </select>
                  </div>

                  <MachineTable
                    data={filteredMachines}
                    onSelectMachine={setSelectedMachine}
                  />
                </section>
              )}

              {/* ======================================================
                  4. PREDICTIVE ALERTS PAGE
                  ====================================================== */}
              {activePage === "Predictive Alerts" && (
                <section className="panel machine-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Flagged Machines & Critical Alerts</h3>
                      <p>Model alerts at the configured threshold for the current replay snapshot.</p>
                    </div>
                    <span className="panel-tag" style={{ background: "#fee2e2", color: "#b91c1c" }}>
                      {activeAlerts.length} ACTIVE ALERTS
                    </span>
                  </div>

                  {activeAlerts.length > 0 ? (
                    <MachineTable
                      data={activeAlerts}
                      onSelectMachine={setSelectedMachine}
                      showExplanation
                    />
                  ) : (
                    <p className="empty-state">
                      No current replay alerts. Start or advance the historical replay to generate model predictions.
                    </p>
                  )}
                </section>
              )}

              {/* ======================================================
                  5. MAINTENANCE PLANNING PAGE
                  ====================================================== */}
              {activePage === "Maintenance" && (
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Capacity-Limited Maintenance Plan</h3>
                      <p>Replay-based planning proposal; it does not dispatch technicians or verify plant availability.</p>
                    </div>
                    <button type="button" className="view-button" onClick={handleOpenReport}>
                      📄 Generate Dispatch Work Orders
                    </button>
                  </div>

                  <div className="stats-grid" style={{ marginBottom: "20px" }}>
                    <StatCard
                      title="Urgent Work Orders"
                      value={simulation ? displayedMachines.filter((m) => m.maintenance_category === "URGENT").length : "—"}
                      note="Current replay CRITICAL risk-band count"
                      icon="⚠"
                      color="red"
                    />
                    <StatCard
                      title="Preventive Work Orders"
                      value={simulation ? displayedMachines.filter((m) => m.maintenance_category === "PREVENTIVE").length : "—"}
                      note="Current replay HIGH/MEDIUM risk-band count"
                      icon="⚙"
                      color="orange"
                    />
                    <StatCard
                      title="Technician Capacity"
                      value={`${simulationCapacity} units/day`}
                      note={`Replay plan horizon: ${simulation?.capacity?.estimated_planning_days ?? "—"} days`}
                      icon="⏱"
                      color="blue"
                    />
                    <StatCard
                      title="Assumed Repair Duration"
                      value={simulation ? `${displayedSchedule.reduce((sum, item) => sum + Number(item.predicted_downtime_hours || 0), 0).toFixed(0)} hrs` : "—"}
                      note="Component-duration assumptions; not downtime avoided"
                      icon="🛡"
                      color="green"
                    />
                  </div>

                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Proposed Slot</th>
                          <th>Rank</th>
                          <th>Machine ID</th>
                          <th>Line</th>
                          <th>Model</th>
                          <th>Priority</th>
                          <th>Maintenance Type</th>
                          <th>RUL (Hrs)</th>
                          <th>Assumed Repair Duration</th>
                          <th>Target Component</th>
                          <th>Feasibility</th>
                          <th>Recommended Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {displayedSchedule.length === 0 ? (
                          <tr><td colSpan="12">{simulation ? "No non-LOW risk machines require a proposed inspection at this snapshot." : "No model-generated schedule is shown until a replay snapshot is active."}</td></tr>
                        ) : (
                          displayedSchedule.map((item, idx) => (
                            <tr
                              key={`${item.machineID}-${idx}`}
                              className="clickable-row"
                              onClick={() => {
                                const full = displayedMachines.find((x) => x.machineID === item.machineID) || item;
                                setSelectedMachine(full);
                              }}
                            >
                              <td><strong>{item.allocation_label || `Day ${Math.floor(idx / simulationCapacity) + 1}`} &bull; Slot {item.daily_slot || (idx % simulationCapacity) + 1}</strong></td>
                              <td>#{item.inspection_rank || idx + 1}</td>
                              <td>Machine {item.machineID}</td>
                              <td>{item.production_line || "—"}</td>
                              <td>{item.model}</td>
                              <td>
                                <span className={`priority ${item.priority.toLowerCase()}`}>
                                  {item.priority}
                                </span>
                              </td>
                              <td>
                                <span className={`maint-tag ${getMaintTagClass(item.maintenance_category)}`}>
                                  {item.maintenance_category || "PREVENTIVE"}
                                </span>
                              </td>
                              <td><strong>{item.predicted_rul_hours == null ? "—" : `${Number(item.predicted_rul_hours).toFixed(1)} hrs`}</strong></td>
                              <td>{item.predicted_downtime_hours == null ? "—" : `${Number(item.predicted_downtime_hours).toFixed(1)} hrs`}</td>
                              <td>{item.target_component || item.component_inspection_focus || "—"}</td>
                              <td title={item.scheduling_reason}>{item.status || "Not evaluated"}</td>
                              <td>{item.recommended_action || "—"}</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}

              {/* ======================================================
                  6. OPERATIONAL TRIAL: MEASURED DOWNTIME REDUCTION PAGE
                  ====================================================== */}
              {activePage === "Operational Trial" && (
                <>
                  <section className="stats-grid">
                    <StatCard
                      title="Scenario Downtime Difference"
                      value={downtimeTrial?.summary?.downtime_reduction_percentage == null ? "—" : `${downtimeTrial.summary.downtime_reduction_percentage}%`}
                      note="Calculated from assumed repair durations; not measured savings"
                      icon="📉"
                      color="green"
                    />
                    <StatCard
                      title="Scenario Cost Difference"
                      value={downtimeTrial?.summary?.financial_savings_usd == null ? "—" : `$${(downtimeTrial.summary.financial_savings_usd / 1000000).toFixed(2)}M`}
                      note={`Assumed rate: $${downtimeTrial?.summary?.hourly_cost_basis_usd ?? "—"}/hr`}
                      icon="💰"
                      color="blue"
                    />
                    <StatCard
                      title="Operational MTBF Effect"
                      value="Not evaluated"
                      note="No observed deployment outcomes are available"
                      icon="⏱"
                      color="orange"
                    />
                    <StatCard
                      title="Trial Scope Evaluated"
                      value={`${downtimeTrial?.trial_scope?.total_historical_failure_events ?? "—"} Events`}
                      note={`${downtimeTrial?.trial_scope?.total_monitored_machines ?? "—"} machines in the recorded dataset`}
                      icon="🏭"
                      color="blue"
                    />
                  </section>

                  {/* COMPARATIVE TRIAL PANEL */}
                  <section className="panel" style={{ marginTop: "20px" }}>
                    <div className="panel-heading">
                      <div>
                        <h3>Illustrative Repair-Duration Scenario</h3>
                        <p>Historical event counts are combined with assumed repair durations; this is not an operational trial.</p>
                      </div>
                      <span className="panel-tag" style={{ background: "#fef3c7", color: "#92400e" }}>
                        NOT VALIDATED
                      </span>
                    </div>

                    <div className="charts-grid" style={{ marginBottom: "20px" }}>
                      <div className="panel" style={{ border: "1px solid #fee2e2", background: "#fffafb" }}>
                        <h4 style={{ color: "#b91c1c", margin: "0 0 8px" }}>Assumed reactive repair scenario</h4>
                        <p style={{ fontSize: "12px", color: "#64748b" }}>
                          Assumes component repair time × 1.5 plus seven hours of logistics overhead. These durations are not measured from operations.
                        </p>
                        <h2 style={{ color: "#dc2626", margin: "10px 0 4px" }}>
                          {downtimeTrial?.summary?.reactive_unplanned_downtime_hours?.toLocaleString() ?? "—"} Hours
                        </h2>
                        <small style={{ color: "#7f1d1d" }}>Scenario total only</small>
                      </div>

                      <div className="panel" style={{ border: "1px solid #bbf7d0", background: "#f0fdf4" }}>
                        <h4 style={{ color: "#15803d", margin: "0 0 8px" }}>Assumed planned repair scenario</h4>
                        <p style={{ fontSize: "12px", color: "#64748b" }}>
                          Assumes component repair time × 0.9. The system does not demonstrate that alerts prevent failures or make all repairs planned.
                        </p>
                        <h2 style={{ color: "#16a34a", margin: "10px 0 4px" }}>
                          {downtimeTrial?.summary?.predictive_planned_downtime_hours?.toLocaleString() ?? "—"} Hours
                        </h2>
                        <small style={{ color: "#14532d" }}>Scenario total only</small>
                      </div>
                    </div>

                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Component</th>
                            <th>Subsystem Name</th>
                            <th>Failure Events</th>
                            <th>Assumed reactive duration</th>
                            <th>Assumed planned duration</th>
                            <th>Scenario difference</th>
                            <th>Scenario cost difference ($)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {downtimeTrial?.breakdown_by_component?.reactive?.map((item, idx) => {
                            const pro = downtimeTrial?.breakdown_by_component?.proactive?.[idx] || {};
                            const saved = (item.total_unplanned_downtime_hours - pro.total_planned_downtime_hours).toFixed(1);
                            const dollars = Math.round((item.total_unplanned_downtime_hours - pro.total_planned_downtime_hours) * (downtimeTrial?.assumptions?.downtime_cost_per_hour_usd || 0)).toLocaleString();
                            return (
                              <tr key={item.component}>
                                <td><strong>{item.component}</strong></td>
                                <td>{item.component_name}</td>
                                <td>{item.failure_events}</td>
                                <td style={{ color: "#dc2626" }}>{item.unit_unplanned_downtime_hours} hrs ({item.total_unplanned_downtime_hours}h total)</td>
                                <td style={{ color: "#16a34a" }}>{pro.unit_planned_downtime_hours} hrs ({pro.total_planned_downtime_hours}h total)</td>
                                <td><strong style={{ color: "#16a34a" }}>+{saved} hrs</strong></td>
                                <td><strong>${dollars}</strong></td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </>
              )}

              {/* ======================================================
                  7. ANALYTICS & HISTORICAL FAILURE TRENDS PAGE
                  ====================================================== */}
              {activePage === "Analytics" && (
                <>
                  <section className="stats-grid">
                    <StatCard
                      title="Total Historical Failures"
                      value={historicalTrends?.total_failures ?? "—"}
                      note="Count in the historical failure dataset"
                      icon="✓"
                      color="blue"
                    />
                    <StatCard
                      title="Estimated Historical Downtime"
                      value={`${historicalTrends?.total_downtime_hours ?? "—"} hrs`}
                      note="Derived from assumed duration per failure component"
                      icon="⏱"
                      color="orange"
                    />
                    <StatCard
                      title="Average Early Warning"
                      value={failureMatches.length ? `${averageWarningHours.toFixed(1)} hrs` : "Not available"}
                      note="Lead time before matched failure"
                      icon="◷"
                      color="green"
                    />
                    <StatCard
                      title="Early Warning Precision"
                      value={failureMatches.length ? `${(averagePeakProbability * 100).toFixed(1)}%` : "Not available"}
                      note="Mean peak probability in warning window"
                      icon="↗"
                      color="red"
                    />
                  </section>

                  {/* MONTHLY FAILURE TRENDS CHART (2015) */}
                  <section className="panel" style={{ marginTop: "20px" }}>
                    <div className="panel-heading">
                      <div>
                        <h3>Historical Failure Trends by Month & Component (2015)</h3>
                        <p>Real historical failure events from `cleaned_failures.csv` aggregated across all 12 months</p>
                      </div>
                      <span className="panel-tag">REAL BENCHMARK DATA</span>
                    </div>

                    <div className="chart" style={{ height: "340px" }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={historicalTrends?.monthly_trends || []}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e9edf3" />
                          <XAxis dataKey="month" axisLine={false} tickLine={false} />
                          <YAxis axisLine={false} tickLine={false} />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="comp1" name="Comp 1 (Electrical)" fill="#3b82f6" stackId="a" />
                          <Bar dataKey="comp2" name="Comp 2 (Drive Shaft)" fill="#f97316" stackId="a" />
                          <Bar dataKey="comp3" name="Comp 3 (Pressure Pump)" fill="#eab308" stackId="a" />
                          <Bar dataKey="comp4" name="Comp 4 (Bearing)" fill="#ef4444" stackId="a" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </section>

                  {/* HISTORICAL DOWNTIME TREND */}
                  <section className="charts-grid" style={{ marginTop: "20px" }}>
                    <div className="panel">
                      <div className="panel-heading">
                        <div>
                          <h3>Estimated Historical Downtime</h3>
                          <p>Calculated from failure counts and assumed component durations; no avoided downtime is inferred.</p>
                        </div>
                      </div>
                      <div className="chart">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={historicalTrends?.monthly_trends || []}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e9edf3" />
                            <XAxis dataKey="month" axisLine={false} tickLine={false} />
                            <YAxis axisLine={false} tickLine={false} />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="downtime_hours" name="Estimated Downtime (hrs)" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="panel">
                      <div className="panel-heading">
                        <div>
                          <h3>Early Warning Lead Time Distribution</h3>
                          <p>Distribution of earliest alert warnings before failure</p>
                        </div>
                      </div>
                      <div className="chart">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={warningLeadTimeData}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e9edf3" />
                            <XAxis dataKey="range" axisLine={false} tickLine={false} />
                            <YAxis allowDecimals={false} axisLine={false} tickLine={false} />
                            <Tooltip />
                            <Bar dataKey="count" name="Matched Failure Events" fill="#16a085" radius={[6, 6, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </section>

                  {/* COMPONENT PROFILE */}
                  <section className="panel" style={{ marginTop: "20px" }}>
                    <div className="panel-heading">
                      <div>
                        <h3>Component Failure & Downtime Impact Matrix</h3>
                        <p>Unit downtime hours and severity weights for each manufacturing component</p>
                      </div>
                    </div>

                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Component</th>
                            <th>Subsystem Name</th>
                            <th>Total Failures</th>
                            <th>Failure Share</th>
                            <th>Assumed Unit Duration</th>
                            <th>Estimated Total Duration</th>
                            <th>Risk Severity Weight</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(historicalTrends?.component_summary || {}).map(([key, info]) => (
                            <tr key={key}>
                              <td><strong>{key}</strong></td>
                              <td>{info.name}</td>
                              <td>{info.failure_count}</td>
                              <td>{info.share_percent}%</td>
                              <td>{info.unit_downtime_hours} hrs</td>
                              <td><strong>{info.total_downtime_hours} hrs</strong></td>
                              <td>
                                <span className={`maint-tag ${info.severity_weight >= 0.8 ? "maint-urgent" : "maint-preventive"}`}>
                                  {(info.severity_weight * 100).toFixed(0)}% Severity
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </>
              )}

              {/* DISCLAIMER */}
              <p className="data-disclaimer">
                {simulation
                  ? `Production Line Replay Snapshot &bull; ${simulation.timestamp} &bull; Model predictions derived strictly from Azure PdM telemetry.`
                  : "Predictive-maintenance prototype &bull; Historical benchmark data only &bull; No live factory feed."}
              </p>
            </>
          )}
        </section>
      </main>

      {/* MACHINE DETAILS MODAL */}
      {selectedMachine && (
        <MachineDetailsModal
          machine={selectedMachine}
          onClose={() => setSelectedMachine(null)}
        />
      )}

      {/* MAINTENANCE DISPATCH REPORT MODAL */}
      {showReportModal && reportError && (
        <div className="machine-modal-overlay" role="dialog" aria-modal="true">
          <div className="machine-modal">
            <p className="simulation-error">{reportError}</p>
            <button type="button" className="view-button" onClick={() => setShowReportModal(false)}>
              Close
            </button>
          </div>
        </div>
      )}
      {showReportModal && reportData && (
        <ReportModal
          reportData={reportData}
          onClose={() => setShowReportModal(false)}
        />
      )}
    </div>
  );
}

// ======================================================
// MACHINE TABLE COMPONENT
// ======================================================

function MachineTable({ data, onSelectMachine, showExplanation = false }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Machine ID</th>
            <th>Line</th>
            <th>Model</th>
            <th>Health Score</th>
            <th>Failure Prob (24h)</th>
            <th>RUL (Hours)</th>
            <th>Predicted Downtime</th>
            <th>Maintenance Priority</th>
            <th>Action</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr><td colSpan="11">No machine records match the active criteria.</td></tr>
          ) : (
            data.map((m) => (
              <tr
                key={m.machineID}
                className="clickable-row"
                onClick={() => onSelectMachine(m)}
              >
                <td><strong>#{m.rank || m.machineID}</strong></td>
                <td>Machine {m.machineID}</td>
                <td>{m.production_line || "—"}</td>
                <td>{m.model}</td>
                <td>
                  {m.health_score != null && Number.isFinite(Number(m.health_score)) ? (
                    <span className={`health-pill ${getHealthClass(Number(m.health_score))}`}>
                      {Number(m.health_score).toFixed(0)}%
                    </span>
                  ) : "Not measured"}
                </td>
                <td>
                  {m.failure_probability_24h != null && Number.isFinite(Number(m.failure_probability_24h)) ? (
                    <strong style={{ color: PRIORITY_COLORS[m.priority] || "#334155" }}>
                      {(Number(m.failure_probability_24h) * 100).toFixed(1)}%
                    </strong>
                  ) : "Not predicted"}
                </td>
                <td>
                  {m.predicted_rul_hours != null && Number.isFinite(Number(m.predicted_rul_hours)) ? (
                    <span style={{ fontWeight: m.predicted_rul_hours <= 24 ? 700 : 500, color: m.predicted_rul_hours <= 24 ? "#dc2626" : "#334155" }}>
                      {Number(m.predicted_rul_hours).toFixed(1)} hrs
                    </span>
                  ) : "Not predicted"}
                </td>
                <td>{m.predicted_downtime_hours != null && Number.isFinite(Number(m.predicted_downtime_hours)) ? `${Number(m.predicted_downtime_hours).toFixed(1)} hrs` : "Not estimated"}</td>
                <td>
                  <span className={`maint-tag ${getMaintTagClass(m.maintenance_category)}`}>
                    {m.maintenance_category || "—"}
                  </span>
                </td>
                <td>
                  <small style={{ color: "#475569" }}>
                    {m.recommended_action || "No current recommendation"}
                  </small>
                </td>
                <td>
                  <button
                    type="button"
                    className="details-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectMachine(m);
                    }}
                  >
                    Diagnose →
                  </button>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

// ======================================================
// MACHINE DETAILS & DIAGNOSTICS MODAL
// ======================================================

function MachineDetailsModal({ machine, onClose }) {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadDetails() {
      try {
        setLoading(true);
        setError("");
        const query = machine.snapshot_datetime ? `?as_of=${encodeURIComponent(machine.snapshot_datetime)}` : "";
        const res = await fetch(`${API_BASE}/api/machines/${machine.machineID}/details${query}`);
        const result = await res.json();
        if (!res.ok) throw new Error(result.detail || "Could not load machine diagnostics.");
        setDetails(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadDetails();
  }, [machine.machineID, machine.snapshot_datetime]);

  const p = details?.prediction;
  const sensor = details?.sensor_readings || machine.sensor_readings || {};
  const explanation = p?.explanation;
  const modelContributions = explanation?.model_contributions || [];

  return (
    <div className="machine-modal-overlay" onClick={onClose}>
      <div className="machine-modal" onClick={(e) => e.stopPropagation()} role="dialog">
        <div className="machine-modal-header">
          <div>
            <span className="modal-eyebrow">PREDICTIVE MAINTENANCE DIAGNOSTICS</span>
            <h2>Machine {machine.machineID} &bull; {details?.model || machine.model}</h2>
            <p>{details?.production_line || machine.production_line} &bull; Operating Age: {details?.age || machine.age} Years</p>
          </div>
          <button type="button" className="modal-close" onClick={onClose}>&times;</button>
        </div>

        <div className="machine-modal-body">
          {loading && <p>Loading sensor readings and multi-model diagnostics...</p>}
          {error && <p className="simulation-error">{error}</p>}

          {!loading && !error && details && (
            <>
              {/* FIVE PILLARS SUMMARY */}
              <div className="report-meta-grid" style={{ marginBottom: "20px" }}>
                <div className="report-meta-item">
                  <span>Machine Health Score</span>
                  <strong style={{ color: p?.health_score >= 80 ? "#16a34a" : p?.health_score >= 50 ? "#ea580c" : "#dc2626" }}>
                    {p?.health_score}%
                  </strong>
                  <small style={{ fontSize: "11px", color: "#64748b" }}>                  {p?.composite_health_index?.health_grade ?? "Not calculated"}</small>
                </div>
                <div className="report-meta-item">
                  <span>24h Failure Probability</span>
                  <strong style={{ color: "#dc2626" }}>
                    {p?.failure_probability_percentage}%
                  </strong>
                  <small style={{ fontSize: "11px", color: "#64748b" }}>Risk Severity: {p?.risk_severity_score}/100</small>
                </div>
                <div className="report-meta-item">
                  <span>RUL Model Output</span>
                  <strong>{p?.predicted_rul_hours == null ? "Not available" : `${p.predicted_rul_hours} hrs`}</strong>
                  <small style={{ fontSize: "11px", color: "#64748b" }}>No calibrated confidence interval is available.</small>
                </div>
                <div className="report-meta-item">
                  <span>Assumption-Based Repair Duration</span>
                  <strong>{p?.downtime_breakdown?.total_downtime_hours || p?.predicted_downtime_hours} Hours</strong>
                  <small style={{ fontSize: "11px", color: "#64748b" }}>Scenario cost estimate: ${p?.downtime_breakdown?.estimated_financial_loss_usd?.toLocaleString() ?? "Not available"}</small>
                </div>
              </div>

              {/* COMPOSITE HEALTH INDEX (CHI) FORMAL BREAKDOWN */}
              {p?.composite_health_index?.breakdown && (
                <div style={{ background: "#f1f5f9", padding: "14px", borderRadius: "8px", marginBottom: "16px", border: "1px solid #cbd5e1" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <strong style={{ fontSize: "13px", color: "#1e293b" }}>Project-Defined Composite Health Index Breakdown:</strong>
                    <span style={{ fontSize: "12px", fontWeight: 700, color: "#475569" }}>Base: 100 - {p.composite_health_index.breakdown.total_penalties} pts = {p.health_score}%</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px", fontSize: "12px" }}>
                    <div style={{ background: "white", padding: "8px", borderRadius: "6px", border: "1px solid #e2e8f0" }}>
                      <span style={{ color: "#64748b" }}>Risk Prob Penalty:</span>
                      <strong style={{ display: "block", color: "#dc2626" }}>-{p.composite_health_index.breakdown.risk_probability_penalty} pts</strong>
                    </div>
                    <div style={{ background: "white", padding: "8px", borderRadius: "6px", border: "1px solid #e2e8f0" }}>
                      <span style={{ color: "#64748b" }}>Sensor Anomaly Penalty:</span>
                      <strong style={{ display: "block", color: "#ea580c" }}>-{p.composite_health_index.breakdown.sensor_anomaly_penalty} pts</strong>
                    </div>
                    <div style={{ background: "white", padding: "8px", borderRadius: "6px", border: "1px solid #e2e8f0" }}>
                      <span style={{ color: "#64748b" }}>24h Error Burst Penalty:</span>
                      <strong style={{ display: "block", color: "#d97706" }}>-{p.composite_health_index.breakdown.error_burst_penalty} pts</strong>
                    </div>
                    <div style={{ background: "white", padding: "8px", borderRadius: "6px", border: "1px solid #e2e8f0" }}>
                      <span style={{ color: "#64748b" }}>Overhaul Wear Penalty:</span>
                      <strong style={{ display: "block", color: "#475569" }}>-{p.composite_health_index.breakdown.maintenance_wear_penalty} pts</strong>
                    </div>
                  </div>
                </div>
              )}

              {/* MODEL EXPLANATION (WHY FLAGGED) */}
              <div className="explanation-box">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <strong style={{ color: "#1e3a8a", fontSize: "14px" }}>
                    Model Prediction and Condition Indicators:
                  </strong>
                  {p?.maintenance_guidance && (
                    <span style={{ fontSize: "11px", fontWeight: 700, padding: "3px 8px", borderRadius: "4px", background: "#fee2e2", color: "#991b1b" }}>
                      {p.maintenance_guidance.zone}
                    </span>
                  )}
                </div>
                <p style={{ margin: "8px 0 0", color: "#1e293b", fontSize: "13px", lineHeight: "1.5" }}>
                  {explanation?.summary || p?.recommended_action}
                </p>
                {explanation?.key_drivers?.length > 0 && (
                  <ul className="explanation-drivers-list">
                    {explanation.key_drivers.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                )}
              </div>

              {/* MODEL FEATURE ATTRIBUTIONS */}
              {modelContributions.length > 0 && (
                <div style={{ marginTop: "16px" }}>
                  <h4 style={{ fontSize: "13px", margin: "0 0 8px", color: "#334155" }}>
                    Top model feature attributions (raw log-odds; not causal effects):
                  </h4>
                  <table style={{ width: "100%", fontSize: "12px", borderCollapse: "collapse", border: "1px solid #e2e8f0" }}>
                    <thead>
                      <tr style={{ background: "#f8fafc" }}>
                        <th style={{ padding: "6px 10px", border: "1px solid #e2e8f0", textAlign: "left" }}>Feature</th>
                        <th style={{ padding: "6px 10px", border: "1px solid #e2e8f0", textAlign: "left" }}>Imputed model input</th>
                        <th style={{ padding: "6px 10px", border: "1px solid #e2e8f0", textAlign: "left" }}>Contribution (log-odds)</th>
                        <th style={{ padding: "6px 10px", border: "1px solid #e2e8f0", textAlign: "left" }}>Direction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelContributions.map((feat, idx) => (
                        <tr key={idx} style={{ background: idx % 2 === 0 ? "white" : "#fafafa" }}>
                          <td style={{ padding: "6px 10px", border: "1px solid #e2e8f0" }}><strong>{feat.feature}</strong></td>
                          <td style={{ padding: "6px 10px", border: "1px solid #e2e8f0" }}>{feat.value}</td>
                          <td style={{ padding: "6px 10px", border: "1px solid #e2e8f0" }}>{feat.contribution_log_odds}</td>
                          <td style={{ padding: "6px 10px", border: "1px solid #e2e8f0" }}>{feat.direction} model score</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* SENSOR READINGS (TEMPERATURE, VIBRATION, PRESSURE, VOLT, ROTATE) */}
              <h3 style={{ margin: "24px 0 12px", fontSize: "15px" }}>Historical Snapshot Readings</h3>
              <div className="machine-detail-grid">
                <div className="machine-detail-card">
                  <span>Temperature Proxy (Formula-Derived)</span>
                  <strong style={{ color: sensor.temperature > 85 ? "#dc2626" : "#0f172a" }}>
                    {sensor.temperature == null ? "Not recorded" : `${sensor.temperature} °C`}
                  </strong>
                </div>
                <div className="machine-detail-card">
                  <span>Vibration (mm/s)</span>
                  <strong style={{ color: sensor.vibration > 45 ? "#dc2626" : "#0f172a" }}>
                    {sensor.vibration == null ? "Not recorded" : Number(sensor.vibration).toFixed(2)}
                  </strong>
                </div>
                <div className="machine-detail-card">
                  <span>Pressure (kPa)</span>
                  <strong>{sensor.pressure == null ? "Not recorded" : Number(sensor.pressure).toFixed(2)}</strong>
                </div>
                <div className="machine-detail-card">
                  <span>Rotation Speed (RPM)</span>
                  <strong>                  {sensor.rotate == null ? "Not recorded" : Number(sensor.rotate).toFixed(1)}</strong>
                </div>
                <div className="machine-detail-card">
                  <span>Voltage (V)</span>
                  <strong>                  {sensor.volt == null ? "Not recorded" : Number(sensor.volt).toFixed(1)}</strong>
                </div>
                <div className="machine-detail-card">
                  <span>Cumulative Operating Runtime</span>
                  <strong>{details.runtime_hours == null ? "Not available" : `${Number(details.runtime_hours).toLocaleString()} hrs`}</strong>
                </div>
              </div>

              {/* MAINTENANCE RECOMMENDATION */}
              <div className="machine-recommendation" style={{ marginTop: "20px" }}>
                <h3>Advisory Maintenance Proposal</h3>
                <p>
                  Classification:{" "}
                  <span className={`maint-tag ${getMaintTagClass(p?.maintenance_category)}`}>
                    {p?.maintenance_category || "PREVENTIVE"}
                  </span>
                  {" &bull; "}
                  Target Subsystem: <strong>{p?.predicted_component_name} ({p?.predicted_component})</strong>
                </p>
                <p>
                  Schedule Window: <strong>{p?.recommended_schedule}</strong>
                </p>
                <p>Action: {p?.recommended_action}</p>
              </div>

              {/* HISTORICAL DOWNTIME & FAILURES */}
              <div className="machine-last-alert" style={{ marginTop: "16px" }}>
                <h3>Historical Failure & Maintenance Records</h3>
                <p>
                  Failure events: <strong>{details.historical_failure_count ?? 0}</strong> &bull; Assumption-based downtime estimate: <strong>{details.historical_total_downtime_hours ?? "Not available"} Hours</strong>
                </p>
                <p>
                  Hours Since Last Recorded Maintenance: <strong>{details.runtime_since_maintenance_hours == null ? "Not recorded" : `${details.runtime_since_maintenance_hours} Hours`}</strong>
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ======================================================
// MAINTENANCE DISPATCH REPORT MODAL
// ======================================================

function ReportModal({ reportData, onClose }) {
  if (!reportData) return null;

  return (
    <div className="machine-modal-overlay" onClick={onClose}>
      <div className="report-modal" onClick={(e) => e.stopPropagation()}>
        <div className="report-header">
          <div>
            <span className="modal-eyebrow">REPLAY-BASED PLAN PROPOSAL</span>
            <h2 style={{ margin: "6px 0 2px" }}>Capacity-Limited Maintenance Proposal</h2>
            <small style={{ color: "#64748b" }}>Historical snapshot: {reportData.generated_at}</small>
          </div>
          <div style={{ display: "flex", gap: "10px" }}>
            <button type="button" className="report-print-btn" onClick={() => window.print()}>
              🖨 Print / PDF
            </button>
            <button type="button" className="modal-close" onClick={onClose}>&times;</button>
          </div>
        </div>

        <div className="report-meta-grid">
          <div className="report-meta-item">
            <span>Average Project Health Index</span>
            <strong>{reportData.factory_overview?.average_health_score == null ? "Not measured" : `${reportData.factory_overview.average_health_score}%`}</strong>
          </div>
          <div className="report-meta-item">
            <span>CRITICAL Proposals</span>
            <strong style={{ color: "#dc2626" }}>{reportData.urgent_work_orders?.length || 0}</strong>
          </div>
          <div className="report-meta-item">
            <span>HIGH/MEDIUM Proposals</span>
            <strong style={{ color: "#ea580c" }}>{reportData.preventive_work_orders?.length || 0}</strong>
          </div>
          <div className="report-meta-item">
            <span>Estimated Days</span>
            <strong>{reportData.capacity_settings?.estimated_planning_days} Days</strong>
          </div>
        </div>

        <p>These are model-based proposals, not submitted work orders. Review feasibility, procedures, and plant constraints before acting.</p>
        <h3 style={{ fontSize: "14px", marginTop: "20px" }}>CRITICAL Risk-Band Proposals</h3>
        <table className="report-table">
          <thead>
            <tr>
              <th>Work Order</th>
              <th>Machine ID</th>
              <th>Line</th>
              <th>Risk</th>
              <th>RUL (Hrs)</th>
              <th>Target Subsystem</th>
              <th>Proposed Time Window</th>
              <th>Feasibility</th>
            </tr>
          </thead>
          <tbody>
            {reportData.urgent_work_orders?.length === 0 ? (
              <tr><td colSpan="8">No CRITICAL risk-band proposal at this timestamp.</td></tr>
            ) : (
              reportData.urgent_work_orders?.map((wo) => (
                <tr key={wo.work_order_id}>
                  <td><strong>{wo.work_order_id}</strong></td>
                  <td>Machine {wo.machineID}</td>
                  <td>{wo.production_line}</td>
                  <td style={{ color: "#dc2626", fontWeight: 700 }}>{wo.failure_probability}</td>
                  <td>{wo.predicted_rul_hours}</td>
                  <td>{wo.target_component}</td>
                  <td>{wo.scheduled_slot}</td>
                  <td>{wo.status}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>

        <h3 style={{ fontSize: "14px", marginTop: "24px" }}>HIGH/MEDIUM Inspection Queue</h3>
        <table className="report-table">
          <thead>
            <tr>
              <th>Work Order</th>
              <th>Machine ID</th>
              <th>Line</th>
              <th>RUL</th>
              <th>Target Subsystem</th>
              <th>Proposed Slot</th>
              <th>Feasibility</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {reportData.preventive_work_orders?.map((wo) => (
              <tr key={wo.work_order_id}>
                <td>{wo.work_order_id}</td>
                <td>Machine {wo.machineID}</td>
                <td>{wo.production_line}</td>
                <td>{wo.predicted_rul_hours}</td>
                <td>{wo.target_component}</td>
                <td>{wo.scheduled_slot}</td>
                <td>{wo.status}</td>
                <td>{wo.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ======================================================
// STAT CARD COMPONENT
// ======================================================

function StatCard({ title, value, note, icon, color = "blue" }) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${color}`}>{icon}</div>
      <p>{title}</p>
      <h2>{value}</h2>
      <small>{note}</small>
    </div>
  );
}

export default App;
