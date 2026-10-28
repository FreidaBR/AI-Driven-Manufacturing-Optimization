
import { useEffect, useMemo, useState } from "react";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
} from "recharts";

import "./style.css";

const API_BASE = "http://127.0.0.1:8000";

const PRIORITY_COLORS = {
  CRITICAL: "#ef4444",
  HIGH: "#f97316",
  MEDIUM: "#eab308",
  LOW: "#22c55e",
};

const componentData = [
  { name: "Comp 1", failures: 192 },
  { name: "Comp 2", failures: 259 },
  { name: "Comp 3", failures: 131 },
  { name: "Comp 4", failures: 179 },
];

function App() {
  const [activePage, setActivePage] = useState("Overview");
  const [selectedMachine, setSelectedMachine] = useState(null);

  const [overview, setOverview] = useState(null);
  const [machines, setMachines] = useState([]);
  const [modelHealth, setModelHealth] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [search, setSearch] = useState("");
  const [priorityFilter, setPriorityFilter] = useState("ALL");

  const navigation = [
    { name: "Overview", icon: "◫" },
    { name: "Machines", icon: "▦" },
    { name: "Predictive Alerts", icon: "⚠" },
    { name: "Maintenance", icon: "⚙" },
    { name: "Analytics", icon: "▥" },
  ];

  // Fetch dashboard data from FastAPI
  useEffect(() => {
    async function loadDashboard() {
      try {
        setLoading(true);
        setError("");

        const [
          overviewResponse,
          machinesResponse,
          modelResponse,
        ] = await Promise.all([
          fetch(`${API_BASE}/api/overview`),
          fetch(`${API_BASE}/api/priorities`),
          fetch(`${API_BASE}/api/model/health`),
        ]);

        if (!overviewResponse.ok) {
          throw new Error("Could not load overview data.");
        }

        if (!machinesResponse.ok) {
          throw new Error("Could not load machine priority data.");
        }

        if (!modelResponse.ok) {
          throw new Error("Could not load model health.");
        }

        const overviewData = await overviewResponse.json();
        const machinesData = await machinesResponse.json();
        const modelData = await modelResponse.json();

        setOverview(overviewData);
        setMachines(machinesData.items || []);
        setModelHealth(modelData);
      } catch (err) {
        console.error("Dashboard loading error:", err);

        setError(
          "Unable to connect to the backend. Make sure FastAPI is running."
        );
      } finally {
        setLoading(false);
      }
    }

    loadDashboard();
  }, []);

  // Build chart data from actual API values
  const priorityData = overview
    ? [
        {
          name: "Critical",
          machines: overview.critical,
          color: PRIORITY_COLORS.CRITICAL,
        },
        {
          name: "High",
          machines: overview.high,
          color: PRIORITY_COLORS.HIGH,
        },
        {
          name: "Medium",
          machines: overview.medium,
          color: PRIORITY_COLORS.MEDIUM,
        },
        {
          name: "Low",
          machines: overview.low,
          color: PRIORITY_COLORS.LOW,
        },
      ]
    : [];

  // Search and filter machine records
  const filteredMachines = useMemo(() => {
    return machines.filter((machine) => {
      const searchTerm = search.trim().toLowerCase();

      const matchesSearch =
        String(machine.machineID).includes(searchTerm) ||
        machine.model?.toLowerCase().includes(searchTerm);

      const matchesPriority =
        priorityFilter === "ALL" ||
        machine.priority === priorityFilter;

      return matchesSearch && matchesPriority;
    });
  }, [machines, search, priorityFilter]);

  // Get machines requiring attention
  const alertMachines = machines.filter(
    (machine) => machine.priority !== "LOW"
  );

  return (
    <div className="app">

      {/* SIDEBAR */}
      <aside className="sidebar">

        <div className="brand">
          <div className="brand-icon">M</div>

          <div>
            <h2>
              Manufacture<span>.AI</span>
            </h2>
            <small>INTELLIGENCE PLATFORM</small>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav>
          {navigation.map((item) => (
            <button
              key={item.name}
              className={`nav-item ${
                activePage === item.name ? "active" : ""
              }`}
              onClick={() => setActivePage(item.name)}
            >
              <span>{item.icon}</span>
              {item.name}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">

          <div className="system-status">
            <span className="status-dot" />

            <div>
              <strong>
                {modelHealth?.model_loaded
                  ? "System Online"
                  : loading
                  ? "Checking system"
                  : "System Offline"}
              </strong>

              <small>
                {modelHealth?.model_loaded
                  ? "Model connected"
                  : "Checking model status"}
              </small>
            </div>
          </div>

          <div className="profile">
            <div className="avatar">AI</div>

            <div>
              <strong>Admin Dashboard</strong>
              <small>Manufacturing</small>
            </div>
          </div>

        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main">

        {/* TOPBAR */}
        <header className="topbar">

          <div>
            <span className="breadcrumb">Workspace / </span>
            <strong>{activePage}</strong>
          </div>

          <div className="top-actions">
            <span className="live-indicator">
              <span className="status-dot" />
              {modelHealth?.model_loaded
                ? "Backend Connected"
                : loading
                ? "Connecting..."
                : "Backend Offline"}
            </span>

            <div className="avatar">AI</div>
          </div>

        </header>

        {/* PAGE CONTENT */}
        <section className="content">

          <div className="page-heading">

            <div>
              <p className="eyebrow">
                MANUFACTURING INTELLIGENCE
              </p>

              <h1>{activePage}</h1>

              <p className="subtitle">
                Monitor machine health and anticipate equipment failures.
              </p>
            </div>

            <div className="date-label">
              24-HOUR PREDICTION HORIZON
            </div>

          </div>

          {/* LOADING */}
          {loading && (
            <section className="panel">
              <p>Loading manufacturing data from FastAPI...</p>
            </section>
          )}

          {/* ERROR */}
          {error && (
            <section className="panel">

              <p style={{ color: "#ef4444" }}>
                {error}
              </p>

              <button
                className="view-button"
                onClick={() => window.location.reload()}
              >
                Retry connection
              </button>

            </section>
          )}

          {/* DASHBOARD */}
          {!loading && !error && overview && (
            <>

              {/* OVERVIEW PAGE */}
              {activePage === "Overview" && (
                <>

                  {/* STATISTICS */}
                  <section className="stats-grid">

                    <StatCard
                      title="Total Machines"
                      value={overview.total_machines}
                      note="Across all machine models"
                      icon="▦"
                    />

                    <StatCard
                      title="Critical Priority"
                      value={overview.critical}
                      note="Require highest attention"
                      icon="⚠"
                      color="red"
                    />

                    <StatCard
                      title="High Priority"
                      value={overview.high}
                      note="Schedule inspections"
                      icon="↗"
                      color="orange"
                    />

                    <StatCard
                      title="Alert Episodes"
                      value="395"
                      note="Across validation folds"
                      icon="⌁"
                      color="blue"
                    />

                  </section>

                  {/* CHARTS */}
                  <section className="charts-grid">

                    {/* BAR CHART */}
                    <div className="panel">

                      <div className="panel-heading">

                        <div>
                          <h3>Machine Priority Distribution</h3>

                          <p>
                            Machines grouped by alert episode count
                          </p>
                        </div>

                        <span className="panel-tag">
                          {overview.total_machines} MACHINES
                        </span>

                      </div>

                      <div className="chart">

                        <ResponsiveContainer
                          width="100%"
                          height="100%"
                        >
                          <BarChart data={priorityData}>

                            <CartesianGrid
                              strokeDasharray="3 3"
                              vertical={false}
                              stroke="#e9edf3"
                            />

                            <XAxis
                              dataKey="name"
                              axisLine={false}
                              tickLine={false}
                            />

                            <YAxis
                              axisLine={false}
                              tickLine={false}
                            />

                            <Tooltip />

                            <Bar
                              dataKey="machines"
                              radius={[6, 6, 0, 0]}
                            >
                              {priorityData.map((entry) => (
                                <Cell
                                  key={entry.name}
                                  fill={entry.color}
                                />
                              ))}
                            </Bar>

                          </BarChart>
                        </ResponsiveContainer>

                      </div>
                    </div>

                    {/* PIE CHART */}
                    <div className="panel">

                      <div className="panel-heading">

                        <div>
                          <h3>Priority Composition</h3>
                          <p>Machine classification overview</p>
                        </div>

                      </div>

                      <div className="chart pie-chart">

                        <ResponsiveContainer
                          width="100%"
                          height="100%"
                        >
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
                                <Cell
                                  key={entry.name}
                                  fill={entry.color}
                                />
                              ))}
                            </Pie>

                            <Tooltip />

                          </PieChart>
                        </ResponsiveContainer>

                      </div>

                      <div className="legend">
                        {priorityData.map((item) => (
                          <span key={item.name}>
                            <i
                              style={{ background: item.color }}
                            />
                            {item.name}: {item.machines}
                          </span>
                        ))}
                      </div>

                    </div>

                  </section>

                  {/* BOTTOM GRID */}
                  <section className="bottom-grid">

                    {/* COMPONENT FAILURE CHART */}
                    <div className="panel">

                      <div className="panel-heading">

                        <div>
                          <h3>Historical Component Failures</h3>

                          <p>
                            Failure distribution across components
                          </p>
                        </div>

                      </div>

                      <div className="chart small-chart">

                        <ResponsiveContainer
                          width="100%"
                          height="100%"
                        >
                          <BarChart data={componentData}>

                            <CartesianGrid
                              strokeDasharray="3 3"
                              vertical={false}
                              stroke="#e9edf3"
                            />

                            <XAxis
                              dataKey="name"
                              axisLine={false}
                              tickLine={false}
                            />

                            <YAxis
                              axisLine={false}
                              tickLine={false}
                            />

                            <Tooltip />

                            <Bar
                              dataKey="failures"
                              fill="#5278e8"
                              radius={[5, 5, 0, 0]}
                            />

                          </BarChart>
                        </ResponsiveContainer>

                      </div>

                    </div>

                    {/* MODEL DETAILS */}
                    <div className="panel">

                      <div className="panel-heading">

                        <div>
                          <h3>Prediction Model</h3>
                          <p>Model configuration</p>
                        </div>

                        <span className="model-status">
                          {modelHealth?.model_loaded
                            ? "LOADED"
                            : "OFFLINE"}
                        </span>

                      </div>

                      <div className="model-details">

                        <div>
                          <span>Algorithm</span>
                          <strong>LightGBM</strong>
                        </div>

                        <div>
                          <span>Features</span>
                          <strong>
                            {modelHealth?.feature_count ?? 47}
                          </strong>
                        </div>

                        <div>
                          <span>Prediction target</span>
                          <strong>Failure in 24h</strong>
                        </div>

                        <div>
                          <span>Alert threshold</span>
                          <strong>98%</strong>
                        </div>

                      </div>

                    </div>

                  </section>

                  {/* MACHINE PRIORITY TABLE */}
                  <section className="panel machine-panel">

                    <div className="panel-heading">

                      <div>
                        <h3>Machine Priority Overview</h3>
                        <p>Top-ranked machines from the backend</p>
                      </div>

                      <button
                        className="view-button"
                        onClick={() => setActivePage("Machines")}
                      >
                        View all machines →
                      </button>

                    </div>

                    <MachineTable
                      data={machines.slice(0, 5)}
                      onSelectMachine={setSelectedMachine}
                    />

                  </section>

                </>
              )}

              {/* MACHINES PAGE */}
              {activePage === "Machines" && (
                <section className="panel machine-panel">

                  <div className="panel-heading">

                    <div>
                      <h3>Machine Registry</h3>

                      <p>
                        Showing {filteredMachines.length} of{" "}
                        {machines.length} machines
                      </p>
                    </div>

                  </div>

                  {/* SEARCH AND FILTER */}
                  <div className="machine-filters">

                    <input
                      type="text"
                      placeholder="Search machine ID or model..."
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                    />

                    <select
                      value={priorityFilter}
                      onChange={(e) => setPriorityFilter(e.target.value)}
                    >
                      <option value="ALL">All priorities</option>
                      <option value="CRITICAL">Critical</option>
                      <option value="HIGH">High</option>
                      <option value="MEDIUM">Medium</option>
                      <option value="LOW">Low</option>
                    </select>

                  </div>

                  <MachineTable
                    data={filteredMachines}
                    onSelectMachine={setSelectedMachine}
                  />

                </section>
              )}

              {/* PREDICTIVE ALERTS PAGE */}
              {activePage === "Predictive Alerts" && (
                <section className="panel machine-panel">

                  <div className="panel-heading">

                    <div>
                      <h3>Machines Requiring Attention</h3>

                      <p>
                        Ranked machines with Critical, High, or Medium priority
                      </p>
                    </div>

                  </div>

                  <MachineTable
                    data={alertMachines}
                    onSelectMachine={setSelectedMachine}
                  />

                </section>
              )}

              {/* MAINTENANCE PAGE */}
              {activePage === "Maintenance" && (
                <section className="panel">

                  <h3>Maintenance Planning</h3>

                  <p>
                    Machines requiring inspection:{" "}
                    <strong>
                      {overview.capacity_scenarios?.[0]
                        ?.machines_requiring_inspection ?? 79}
                    </strong>
                  </p>

                  <div className="table-wrap">

                    <table>

                      <thead>
                        <tr>
                          <th>Machines per day</th>
                          <th>Machines to inspect</th>
                          <th>Estimated planning days</th>
                        </tr>
                      </thead>

                      <tbody>
                        {(overview.capacity_scenarios || []).map(
                          (scenario) => (
                            <tr key={scenario.machines_per_day}>
                              <td>{scenario.machines_per_day}</td>
                              <td>
                                {scenario.machines_requiring_inspection}
                              </td>
                              <td>
                                {scenario.estimated_planning_days}
                              </td>
                            </tr>
                          )
                        )}
                      </tbody>

                    </table>

                  </div>

                </section>
              )}

              {/* ANALYTICS PAGE */}
              {activePage === "Analytics" && (
                <section className="panel page-placeholder">

                  <h2>Analytics</h2>

                  <p>
                    Model evaluation, failure matching, and component
                    analytics will be connected here next.
                  </p>

                </section>
              )}

              {/* DATA DISCLAIMER */}
              <p className="data-disclaimer">
                Historical dataset analysis · Not a live factory monitoring feed
              </p>

            </>
          )}

        </section>
      </main>

      {/* MACHINE DETAILS MODAL */}
      {selectedMachine && (
        <MachineDetails
          machine={selectedMachine}
          onClose={() => setSelectedMachine(null)}
        />
      )}

    </div>
  );
}


// ======================================================
// STATISTICS CARD
// ======================================================

function StatCard({
  title,
  value,
  note,
  icon,
  color = "blue",
}) {
  return (
    <div className="stat-card">

      <div className={`stat-icon ${color}`}>
        {icon}
      </div>

      <p>{title}</p>

      <h2>{value}</h2>

      <small>{note}</small>

    </div>
  );
}


// ======================================================
// MACHINE TABLE
// ======================================================


function MachineTable({ data, onSelectMachine }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Machine ID</th>
            <th>Model</th>
            <th>Priority</th>
            <th>Alert Episodes</th>
            <th>Matched Failures</th>
            <th>Highest Probability</th>
            <th>Details</th>
          </tr>
        </thead>

        <tbody>
          {data.length === 0 ? (
            <tr>
              <td colSpan="8">No machines found.</td>
            </tr>
          ) : (
            data.map((machine) => (
              <tr
                key={machine.machineID}
                className="clickable-row"
                onClick={() => onSelectMachine(machine)}
              >
                <td>#{machine.rank}</td>
                <td>Machine {machine.machineID}</td>
                <td>{machine.model}</td>

                <td>
                  <span className={`priority ${machine.priority.toLowerCase()}`}>
                    {machine.priority}
                  </span>
                </td>

                <td>{machine.alert_episodes}</td>
                <td>{machine.matched_failures}</td>

                <td>
                  {(machine.highest_episode_probability * 100).toFixed(2)}%
                </td>

                <td>
                  <button
                    type="button"
                    className="details-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectMachine(machine);
                    }}
                  >
                    View Details
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
// MACHINE DETAILS MODAL
// ======================================================

function MachineDetails({ machine, onClose }) {
  if (!machine) return null;

  const probability = (
    machine.highest_episode_probability * 100
  ).toFixed(3);

  const lastProbability = (
    machine.last_alert_probability * 100
  ).toFixed(3);

  return (
    <div
      className="machine-modal-overlay"
      onClick={onClose}
    >

      <div
        className="machine-modal"
        onClick={(e) => e.stopPropagation()}
      >

        {/* MODAL HEADER */}
        <div className="machine-modal-header">

          <div>
            <span className="modal-eyebrow">
              MACHINE DETAILS
            </span>

            <h2>Machine {machine.machineID}</h2>

            <p>{machine.model.toUpperCase()}</p>
          </div>

          <button
            className="modal-close"
            onClick={onClose}
            aria-label="Close machine details"
          >
            ×
          </button>

        </div>

        <div className="machine-modal-body">

          {/* PRIORITY */}
          <div className="machine-priority-row">

            <span>Current Priority</span>

            <span
              className={`priority ${machine.priority.toLowerCase()}`}
            >
              {machine.priority}
            </span>

          </div>

          {/* MACHINE METRICS */}
          <div className="machine-detail-grid">

            <div className="machine-detail-card">
              <span>Highest Alert Probability</span>
              <strong>{probability}%</strong>
            </div>

            <div className="machine-detail-card">
              <span>Alert Episodes</span>
              <strong>{machine.alert_episodes}</strong>
            </div>

            <div className="machine-detail-card">
              <span>Total Alert Hours</span>
              <strong>{machine.total_alert_hours} hrs</strong>
            </div>

            <div className="machine-detail-card">
              <span>Average Episode Duration</span>
              <strong>
                {Number(machine.average_episode_duration).toFixed(1)} hrs
              </strong>
            </div>

            <div className="machine-detail-card">
              <span>Matched Historical Failures</span>
              <strong>{machine.matched_failures}</strong>
            </div>

            <div className="machine-detail-card">
              <span>Machine Ranking</span>
              <strong>#{machine.rank}</strong>
            </div>

          </div>

          {/* LAST ALERT */}
          <div className="machine-last-alert">

            <h3>Last Alert</h3>

            <p>
              {machine.last_alert_datetime || "No recorded alert"}
            </p>

            <span>
              Last alert probability: {lastProbability}%
            </span>

          </div>

          {/* MAINTENANCE RECOMMENDATION */}
          <div className="machine-recommendation">

            <h3>Recommended Maintenance Action</h3>

            <p>{machine.recommended_action}</p>

          </div>

          {/* DISCLAIMER */}
          <p className="machine-modal-note">
            Historical dataset analysis · Not a live machine status.
          </p>

        </div>

      </div>

    </div>
  );
}

export default App;
