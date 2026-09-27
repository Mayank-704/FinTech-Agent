"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
} from "recharts";
import {
  fetchDashboard,
  sendAgentMessage,
  resumeAction,
  syncPlaid,
  type DashboardData,
  type AgentResponse,
  type PendingAction,
  type TrajectoryPoint,
} from "@/lib/api";

// ─── Demo User ID (replace with auth in production) ──────────────────────────
const DEMO_USER_ID = "00000000-0000-0000-0000-000000000001";

// ─── Utility ──────────────────────────────────────────────────────────────────
function fmt(n: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(n);
}

function fmtDate(d: string) {
  return new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "rgba(10,12,18,0.95)",
      border: "1px solid rgba(0,212,255,0.2)",
      borderRadius: 10,
      padding: "10px 16px",
      fontSize: 13,
    }}>
      <p style={{ color: "rgba(255,255,255,0.5)", marginBottom: 6 }}>{label}</p>
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.color, fontWeight: 600 }}>
          {p.name}: {fmt(p.value)}
        </div>
      ))}
    </div>
  );
}

// ─── Dashboard Page ───────────────────────────────────────────────────────────
export default function AegisDashboard() {
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; content: string; meta?: AgentResponse }>>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [threadId, setThreadId] = useState<string | undefined>(undefined);
  const [ghostTrajectory, setGhostTrajectory] = useState<{ baseline?: TrajectoryPoint[]; shocked?: TrajectoryPoint[] } | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "ledger" | "queue">("overview");
  const [syncing, setSyncing] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // ── Load Dashboard ─────────────────────────────────────────────────────────
  const loadDashboard = useCallback(async () => {
    try {
      const data = await fetchDashboard(DEMO_USER_ID);
      setDashboard(data);
    } catch {
      // Use mock data for demo when backend is not running
      setDashboard(MOCK_DASHBOARD);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadDashboard(); }, [loadDashboard]);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // ── Chat Submit ────────────────────────────────────────────────────────────
  async function handleChat(e: React.FormEvent) {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const userMsg = chatInput.trim();
    setChatInput("");
    setMessages(m => [...m, { role: "user", content: userMsg }]);
    setChatLoading(true);

    try {
      const res = await sendAgentMessage(DEMO_USER_ID, userMsg, threadId);
      setThreadId(res.thread_id);

      let content = "";
      const mathResults = res.response?.math_results || res.math_results;

      if (res.status === "pending_approval") {
        content = `⏸ **Action Requires Approval**\n\n${res.message}\n\nCheck the **Command Queue** tab to authorize or reject.`;
      } else if (mathResults?.type === "shock_simulation") {
        content = `🧮 **Shock Simulation Complete**\n\n` +
          `Baseline End Balance: **${fmt(mathResults.baseline_end_balance as number)}**\n` +
          `Shocked End Balance: **${fmt(mathResults.shocked_end_balance as number)}**\n` +
          `Δ Impact: **${fmt(mathResults.delta as number)}**\n` +
          `Insolvency Risk: **${((mathResults.insolvency_risk as number) * 100).toFixed(1)}%**\n\n` +
          `💡 ${mathResults.recommendation}`;
        if (mathResults.baseline_trajectory && mathResults.shocked_trajectory) {
          setGhostTrajectory({
            baseline: mathResults.baseline_trajectory as TrajectoryPoint[],
            shocked: mathResults.shocked_trajectory as TrajectoryPoint[],
          });
          setActiveTab("ledger");
        }
      } else if (mathResults?.type === "portfolio_query") {
        content = `💼 **Portfolio Overview**\n\n` +
          `Total Balance: **${fmt(mathResults.total_balance as number)}**\n` +
          `Net Daily Flow: **${fmt(mathResults.net_daily_flow as number)}**\n` +
          `Runway: **${mathResults.runway_days ? `${mathResults.runway_days} days` : "Positive"}**\n\n` +
          `Monte Carlo P50: ${fmt(mathResults.monte_carlo?.p50 || 0)}`;
        if (mathResults.trajectory) {
          setGhostTrajectory({ baseline: mathResults.trajectory as TrajectoryPoint[] });
          setActiveTab("ledger");
        }
      } else if (mathResults?.type === "transfer_simulation") {
        content = `🔄 **Transfer Simulation**\n\n` +
          `Feasible: **${mathResults.is_feasible ? "Yes ✓" : "No ✗"}**\n` +
          `From Account New Balance: **${fmt(mathResults.from_new_balance as number)}**\n` +
          `To Account New Balance: **${fmt(mathResults.to_new_balance as number)}**` +
          (mathResults.shortfall ? `\nShortfall: **${fmt(mathResults.shortfall as number)}**` : "");
      } else if (res.response?.clarification) {
        content = `❓ ${res.response?.clarification}`;
      } else {
        content = res.error
          ? `⚠️ Error: ${res.error}`
          : "✅ Analysis complete. Check the Ghost Ledger tab for visualizations.";
      }

      setMessages(m => [...m, { role: "assistant", content, meta: res }]);
      if (res.status === "pending_approval") {
        setActiveTab("queue");
        loadDashboard(); // Refresh action queue
      }
    } catch (err) {
      setMessages(m => [...m, {
        role: "assistant",
        content: "⚠️ Backend offline. Running in demo mode with mock data.",
      }]);
    } finally {
      setChatLoading(false);
    }
  }

  // ── HITL Decision ──────────────────────────────────────────────────────────
  async function handleDecision(action: PendingAction, decision: "APPROVE" | "REJECT") {
    try {
      await resumeAction(action.id, decision, action.id, DEMO_USER_ID);
      setMessages(m => [...m, {
        role: "assistant",
        content: `${decision === "APPROVE" ? "✅ Approved" : "❌ Rejected"}: ${action.type} action has been ${decision.toLowerCase()}d.`,
      }]);
      loadDashboard();
    } catch {
      alert("Failed to process decision. Check console.");
    }
  }

  // ── Plaid Sync ─────────────────────────────────────────────────────────────
  async function handlePlaidSync() {
    setSyncing(true);
    try {
      const res = await syncPlaid(DEMO_USER_ID);
      if (res.degraded_mode) {
        setMessages(m => [...m, { role: "assistant", content: `⚠️ Plaid sync degraded: ${res.error}. Showing last known state.` }]);
      } else {
        setMessages(m => [...m, { role: "assistant", content: `✅ Synced ${res.accounts_synced} accounts and ${res.transactions_synced} transactions.` }]);
      }
      loadDashboard();
    } catch {
      setMessages(m => [...m, { role: "assistant", content: "⚠️ Plaid unavailable. Showing cached data." }]);
    } finally {
      setSyncing(false);
    }
  }

  // ── Ghost Ledger Chart Data ────────────────────────────────────────────────
  const chartData = ghostTrajectory
    ? (ghostTrajectory.baseline || []).map((pt, i) => ({
        date: fmtDate(pt.date),
        "Current Trajectory": pt.balance,
        ...(ghostTrajectory.shocked?.[i]
          ? { "AI Shocked Trajectory": ghostTrajectory.shocked[i].balance }
          : {}),
      }))
    : MOCK_CHART_DATA;

  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh", flexDirection: "column", gap: 16 }}>
        <div style={{ width: 48, height: 48, border: "3px solid rgba(0,212,255,0.3)", borderTop: "3px solid #00d4ff", borderRadius: "50%" }} className="animate-spin-slow" />
        <p style={{ color: "rgba(255,255,255,0.5)", fontFamily: "Space Grotesk" }}>Initializing Aegis...</p>
      </div>
    );
  }

  const d = dashboard || MOCK_DASHBOARD;
  const pendingCount = d.pending_actions?.length || 0;

  return (
    <div style={{ minHeight: "100vh", background: "var(--aegis-black)" }}>
      {/* ── Top Nav ── */}
      <nav style={{
        borderBottom: "1px solid var(--aegis-border)",
        padding: "0 24px",
        height: 64,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "rgba(10,12,18,0.8)",
        backdropFilter: "blur(20px)",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36,
            height: 36,
            background: "var(--gradient-primary)",
            borderRadius: 10,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 18,
          }}>⚡</div>
          <div>
            <span style={{ fontFamily: "Space Grotesk", fontWeight: 700, fontSize: 18, background: "var(--gradient-primary)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              AEGIS
            </span>
            <span style={{ color: "var(--text-muted)", fontSize: 11, display: "block", marginTop: -2, letterSpacing: 2 }}>
              FINANCIAL RISK ORCHESTRATOR
            </span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {d.degraded_mode && (
            <div style={{ background: "var(--aegis-amber-dim)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: 8, padding: "4px 12px", fontSize: 12, color: "#f59e0b" }}>
              ⚠ Degraded Mode
            </div>
          )}
          <button
            id="plaid-sync-btn"
            onClick={handlePlaidSync}
            disabled={syncing}
            style={{
              background: syncing ? "rgba(255,255,255,0.05)" : "rgba(0,212,255,0.08)",
              border: "1px solid rgba(0,212,255,0.25)",
              borderRadius: 8,
              padding: "8px 16px",
              color: "#00d4ff",
              cursor: syncing ? "wait" : "pointer",
              fontSize: 13,
              fontWeight: 500,
              transition: "all 0.2s",
            }}
          >
            {syncing ? "⟳ Syncing..." : "⟳ Sync Plaid"}
          </button>
        </div>
      </nav>

      <div style={{ display: "flex", height: "calc(100vh - 64px)" }}>
        {/* ── Left: Main Content ── */}
        <div style={{ flex: 1, overflow: "auto", padding: 24 }}>
          {/* Balance Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16, marginBottom: 24 }} className="animate-fade-in">
            <StatCard
              label="Total Balance"
              value={fmt(d.total_balance)}
              sub={d.accounts?.length ? `${d.accounts.length} accounts` : "No accounts"}
              accent="#00d4ff"
              icon="💰"
            />
            {d.accounts?.slice(0, 3).map(a => (
              <StatCard
                key={a.id}
                label={a.name}
                value={fmt(a.balance)}
                sub={a.type}
                accent={a.type === "credit" ? "#f43f5e" : "#10b981"}
                icon={a.type === "credit" ? "💳" : "🏦"}
              />
            ))}
          </div>

          {/* Tab Bar */}
          <div style={{ display: "flex", gap: 4, marginBottom: 20, background: "var(--aegis-card)", borderRadius: 12, padding: 4, width: "fit-content", border: "1px solid var(--aegis-border)" }}>
            {([
              { key: "overview", label: "Overview" },
              { key: "ledger", label: "Ghost Ledger" },
              { key: "queue", label: `Command Queue${pendingCount ? ` (${pendingCount})` : ""}` },
            ] as const).map(tab => (
              <button
                key={tab.key}
                id={`tab-${tab.key}`}
                onClick={() => setActiveTab(tab.key)}
                style={{
                  padding: "8px 20px",
                  borderRadius: 8,
                  border: "none",
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 500,
                  transition: "all 0.2s",
                  background: activeTab === tab.key ? "rgba(0,212,255,0.1)" : "transparent",
                  color: activeTab === tab.key ? "#00d4ff" : "rgba(255,255,255,0.4)",
                  position: "relative",
                }}
              >
                {tab.label}
                {tab.key === "queue" && pendingCount > 0 && activeTab !== "queue" && (
                  <span style={{
                    position: "absolute",
                    top: 4,
                    right: 8,
                    width: 7,
                    height: 7,
                    background: "#f43f5e",
                    borderRadius: "50%",
                  }} />
                )}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="animate-fade-in" key={activeTab}>
            {activeTab === "overview" && (
              <TransactionTable transactions={d.recent_transactions || MOCK_TRANSACTIONS} />
            )}
            {activeTab === "ledger" && (
              <GhostLedger data={chartData} />
            )}
            {activeTab === "queue" && (
              <CommandQueue
                actions={d.pending_actions || []}
                onDecision={handleDecision}
              />
            )}
          </div>
        </div>

        {/* ── Right: Chat Panel ── */}
        <div style={{
          width: 380,
          borderLeft: "1px solid var(--aegis-border)",
          display: "flex",
          flexDirection: "column",
          background: "var(--aegis-card)",
        }}>
          {/* Chat Header */}
          <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--aegis-border)" }}>
            <h3 style={{ fontFamily: "Space Grotesk", fontWeight: 700, fontSize: 15 }}>🤖 Aegis Agent</h3>
            <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 2 }}>
              AI financial orchestrator • Zero LLM math
            </p>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflow: "auto", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
            {messages.length === 0 && (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", gap: 16 }}>
                <div style={{ fontSize: 40 }} className="animate-float">⚡</div>
                <p style={{ color: "var(--text-muted)", fontSize: 13, textAlign: "center", lineHeight: 1.6 }}>
                  Ask me anything about your finances.<br />
                  Try: <em>"What if I lose my job?"</em><br />
                  or: <em>"Move $500 to savings"</em>
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%" }}>
                  {["What if I lose my job next month?", "Show my runway", "Simulate a $1000 expense spike"].map(s => (
                    <button
                      key={s}
                      onClick={() => setChatInput(s)}
                      style={{
                        background: "rgba(0,212,255,0.05)",
                        border: "1px solid rgba(0,212,255,0.15)",
                        borderRadius: 8,
                        padding: "8px 14px",
                        color: "rgba(255,255,255,0.6)",
                        cursor: "pointer",
                        fontSize: 12,
                        textAlign: "left",
                        transition: "all 0.2s",
                      }}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((msg, i) => (
              <ChatBubble key={i} role={msg.role} content={msg.content} />
            ))}
            {chatLoading && (
              <div style={{ display: "flex", gap: 6, alignItems: "center", color: "var(--text-muted)", fontSize: 13 }}>
                <div style={{
                  width: 16, height: 16,
                  border: "2px solid rgba(0,212,255,0.3)",
                  borderTop: "2px solid #00d4ff",
                  borderRadius: "50%",
                }} className="animate-spin-slow" />
                Analyzing...
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Chat Input */}
          <form onSubmit={handleChat} style={{ padding: "12px 20px", borderTop: "1px solid var(--aegis-border)" }}>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                id="chat-input"
                value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                placeholder="Ask Aegis..."
                disabled={chatLoading}
                style={{
                  flex: 1,
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 10,
                  padding: "10px 14px",
                  color: "var(--text-primary)",
                  fontSize: 13,
                  outline: "none",
                  transition: "border-color 0.2s",
                  fontFamily: "Inter",
                }}
                onFocus={e => (e.target.style.borderColor = "rgba(0,212,255,0.4)")}
                onBlur={e => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
              />
              <button
                id="chat-send-btn"
                type="submit"
                disabled={chatLoading || !chatInput.trim()}
                style={{
                  width: 40,
                  height: 40,
                  background: chatInput.trim() ? "var(--gradient-primary)" : "rgba(255,255,255,0.05)",
                  border: "none",
                  borderRadius: 10,
                  cursor: chatInput.trim() ? "pointer" : "default",
                  fontSize: 16,
                  transition: "all 0.2s",
                  flexShrink: 0,
                }}
              >
                ➤
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

// ─── Sub-Components ───────────────────────────────────────────────────────────

function StatCard({ label, value, sub, accent, icon }: { label: string; value: string; sub: string; accent: string; icon: string }) {
  return (
    <div style={{
      background: "var(--aegis-card)",
      border: "1px solid var(--aegis-border)",
      borderRadius: 16,
      padding: "20px 24px",
      position: "relative",
      overflow: "hidden",
      transition: "border-color 0.2s, transform 0.2s",
      cursor: "default",
    }}
      onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.borderColor = `${accent}44`; (e.currentTarget as HTMLDivElement).style.transform = "translateY(-2px)"; }}
      onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.borderColor = "var(--aegis-border)"; (e.currentTarget as HTMLDivElement).style.transform = "translateY(0)"; }}
    >
      <div style={{ position: "absolute", top: 0, right: 0, width: 80, height: 80, background: `radial-gradient(circle, ${accent}15, transparent 70%)` }} />
      <div style={{ fontSize: 24, marginBottom: 8 }}>{icon}</div>
      <div style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 500, letterSpacing: 0.5, textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 700, fontFamily: "Space Grotesk", color: accent }}>{value}</div>
      <div style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 4, textTransform: "capitalize" }}>{sub}</div>
    </div>
  );
}

function GhostLedger({ data }: { data: Array<Record<string, number | string>> }) {
  const hasShock = data.some(d => d["AI Shocked Trajectory"] !== undefined);
  return (
    <div style={{ background: "var(--aegis-card)", border: "1px solid var(--aegis-border)", borderRadius: 16, padding: 24 }}>
      <div style={{ marginBottom: 20 }}>
        <h3 style={{ fontFamily: "Space Grotesk", fontWeight: 700, fontSize: 18, marginBottom: 4 }}>
          Ghost Ledger Simulation
        </h3>
        <p style={{ color: "var(--text-secondary)", fontSize: 13 }}>
          {hasShock
            ? "Comparing current trajectory vs AI shock simulation"
            : "Cash flow trajectory – ask Aegis to simulate a shock to see the comparison"}
        </p>
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorCurrent" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorShocked" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis dataKey="date" stroke="rgba(255,255,255,0.2)" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
          <YAxis stroke="rgba(255,255,255,0.2)" tick={{ fontSize: 11 }} tickFormatter={v => `$${(v / 1000).toFixed(1)}k`} tickLine={false} axisLine={false} />
          <Tooltip content={<ChartTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} />
          <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" label={{ value: "Zero", fill: "rgba(255,255,255,0.3)", fontSize: 11 }} />
          <Area type="monotone" dataKey="Current Trajectory" stroke="#00d4ff" strokeWidth={2.5} fill="url(#colorCurrent)" dot={false} activeDot={{ r: 5, fill: "#00d4ff" }} />
          {hasShock && (
            <Area type="monotone" dataKey="AI Shocked Trajectory" stroke="#f43f5e" strokeWidth={2.5} strokeDasharray="6 3" fill="url(#colorShocked)" dot={false} activeDot={{ r: 5, fill: "#f43f5e" }} />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function TransactionTable({ transactions }: { transactions: typeof MOCK_TRANSACTIONS }) {
  return (
    <div style={{ background: "var(--aegis-card)", border: "1px solid var(--aegis-border)", borderRadius: 16, overflow: "hidden" }}>
      <div style={{ padding: "20px 24px", borderBottom: "1px solid var(--aegis-border)" }}>
        <h3 style={{ fontFamily: "Space Grotesk", fontWeight: 700, fontSize: 16 }}>Recent Transactions</h3>
        <p style={{ color: "var(--text-secondary)", fontSize: 12, marginTop: 2 }}>Immutable ledger · append-only</p>
      </div>
      <div style={{ maxHeight: 420, overflow: "auto" }}>
        {transactions.slice(0, 20).map((t, i) => (
          <div
            key={t.id || i}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "14px 24px",
              borderBottom: "1px solid rgba(255,255,255,0.03)",
              transition: "background 0.15s",
            }}
            onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.02)")}
            onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{
                width: 36,
                height: 36,
                borderRadius: 10,
                background: t.amount > 0 ? "var(--aegis-rose-dim)" : "var(--aegis-emerald-dim)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 14,
              }}>
                {t.amount > 0 ? "↑" : "↓"}
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 500 }}>{t.description}</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2, display: "flex", gap: 8 }}>
                  <span>{fmtDate(t.date)}</span>
                  {t.category && <span style={{ color: "rgba(0,212,255,0.6)" }}>· {t.category}</span>}
                  {t.is_recurring && <span style={{ color: "rgba(124,58,237,0.8)" }}>· recurring</span>}
                </div>
              </div>
            </div>
            <div style={{
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 13,
              fontWeight: 600,
              color: t.amount > 0 ? "#f43f5e" : "#10b981",
            }}>
              {t.amount > 0 ? "-" : "+"}{fmt(Math.abs(t.amount))}
            </div>
          </div>
        ))}
        {transactions.length === 0 && (
          <div style={{ padding: "48px 24px", textAlign: "center", color: "var(--text-muted)" }}>
            No transactions. Sync Plaid to load data.
          </div>
        )}
      </div>
    </div>
  );
}

function CommandQueue({ actions, onDecision }: { actions: PendingAction[]; onDecision: (a: PendingAction, d: "APPROVE" | "REJECT") => void }) {
  if (actions.length === 0) {
    return (
      <div style={{
        background: "var(--aegis-card)",
        border: "1px solid var(--aegis-border)",
        borderRadius: 16,
        padding: "64px 24px",
        textAlign: "center",
        color: "var(--text-muted)",
      }}>
        <div style={{ fontSize: 40, marginBottom: 16 }}>✅</div>
        <p style={{ fontSize: 14, fontWeight: 500 }}>No pending actions</p>
        <p style={{ fontSize: 12, marginTop: 8 }}>Ask Aegis to transfer funds or reallocate to see actions here.</p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {actions.map(action => (
        <div
          key={action.id}
          style={{
            background: "var(--aegis-card)",
            border: "1px solid rgba(245,158,11,0.3)",
            borderRadius: 16,
            padding: 24,
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg, #f59e0b, #f43f5e)" }} />
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16 }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span style={{ fontSize: 18 }}>⏸</span>
                <h4 style={{ fontFamily: "Space Grotesk", fontWeight: 700, fontSize: 15 }}>
                  Pending: {action.type.replace("_", " ")}
                </h4>
              </div>
              <span style={{
                background: "var(--aegis-amber-dim)",
                border: "1px solid rgba(245,158,11,0.25)",
                borderRadius: 6,
                padding: "2px 10px",
                fontSize: 11,
                color: "#f59e0b",
                fontWeight: 600,
              }}>
                AWAITING HUMAN APPROVAL
              </span>
            </div>
          </div>

          {/* Payload */}
          <div style={{
            background: "rgba(0,0,0,0.3)",
            borderRadius: 10,
            padding: 14,
            marginBottom: 16,
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 11,
            color: "rgba(255,255,255,0.5)",
            lineHeight: 1.6,
            maxHeight: 100,
            overflow: "auto",
          }}>
            {JSON.stringify(action.payload, null, 2)}
          </div>

          {/* Simulation Preview */}
          {action.simulation_result && (
            <div style={{ background: "rgba(0,212,255,0.05)", border: "1px solid rgba(0,212,255,0.1)", borderRadius: 10, padding: 14, marginBottom: 16 }}>
              <p style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 8 }}>SIMULATION PREVIEW</p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {Object.entries(action.simulation_result || {}).filter(([k, v]) => typeof v === "number").slice(0, 4).map(([k, v]) => (
                  <div key={k}>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase" }}>{k.replace(/_/g, " ")}</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "#00d4ff", fontFamily: "JetBrains Mono" }}>
                      {typeof v === "number" && k.includes("balance") ? fmt(v as number) : String(v)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Decision Buttons */}
          <div style={{ display: "flex", gap: 12 }}>
            <button
              id={`approve-${action.id}`}
              onClick={() => onDecision(action, "APPROVE")}
              style={{
                flex: 1,
                padding: "12px 24px",
                background: "linear-gradient(135deg, #10b981, #059669)",
                border: "none",
                borderRadius: 12,
                color: "white",
                fontWeight: 700,
                fontSize: 14,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                transition: "transform 0.15s, box-shadow 0.15s",
                boxShadow: "0 4px 20px rgba(16,185,129,0.2)",
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.transform = "translateY(-1px)"; (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 6px 24px rgba(16,185,129,0.35)"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.transform = "translateY(0)"; (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 4px 20px rgba(16,185,129,0.2)"; }}
            >
              ✓ Authorize
            </button>
            <button
              id={`reject-${action.id}`}
              onClick={() => onDecision(action, "REJECT")}
              style={{
                flex: 1,
                padding: "12px 24px",
                background: "linear-gradient(135deg, #f43f5e, #e11d48)",
                border: "none",
                borderRadius: 12,
                color: "white",
                fontWeight: 700,
                fontSize: 14,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                transition: "transform 0.15s, box-shadow 0.15s",
                boxShadow: "0 4px 20px rgba(244,63,94,0.2)",
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.transform = "translateY(-1px)"; (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 6px 24px rgba(244,63,94,0.35)"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.transform = "translateY(0)"; (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 4px 20px rgba(244,63,94,0.2)"; }}
            >
              ✗ Reject
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function ChatBubble({ role, content }: { role: "user" | "assistant"; content: string }) {
  const isUser = role === "user";
  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", animation: "fadeIn 0.3s ease" }}>
      <div style={{
        maxWidth: "85%",
        padding: "10px 14px",
        borderRadius: isUser ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
        background: isUser ? "rgba(0,212,255,0.12)" : "rgba(255,255,255,0.05)",
        border: `1px solid ${isUser ? "rgba(0,212,255,0.2)" : "rgba(255,255,255,0.06)"}`,
        fontSize: 13,
        lineHeight: 1.6,
        color: "var(--text-primary)",
        whiteSpace: "pre-wrap",
      }}>
        {content.split("**").map((part, i) =>
          i % 2 === 0 ? part : <strong key={i} style={{ color: "#00d4ff" }}>{part}</strong>
        )}
      </div>
    </div>
  );
}

// ─── Mock Data (shown when backend is offline) ────────────────────────────────

const MOCK_DASHBOARD: DashboardData = {
  status: "ok",
  degraded_mode: false,
  total_balance: 24750.00,
  accounts: [
    { id: "1", name: "Chase Checking", balance: 8250.50, type: "depository" },
    { id: "2", name: "Chase Savings", balance: 12500.00, type: "depository" },
    { id: "3", name: "Amex Platinum", balance: -3200.00, type: "credit" },
    { id: "4", name: "Fidelity Brokerage", balance: 7200.00, type: "investment" },
  ],
  recent_transactions: [
    { id: "t1", amount: 85.00, date: "2026-09-27", description: "Netflix, Spotify", category: "Entertainment", is_recurring: true },
    { id: "t2", amount: 420.00, date: "2026-09-26", description: "Whole Foods Market", category: "Groceries", is_recurring: false },
    { id: "t3", amount: -6500.00, date: "2026-09-25", description: "Employer Payroll", category: "Income", is_recurring: true },
    { id: "t4", amount: 1850.00, date: "2026-09-24", description: "Rent – October", category: "Housing", is_recurring: true },
    { id: "t5", amount: 68.40, date: "2026-09-23", description: "Shell Gas Station", category: "Transport", is_recurring: false },
    { id: "t6", amount: 245.00, date: "2026-09-22", description: "Electric & Internet Bundle", category: "Utilities", is_recurring: true },
    { id: "t7", amount: 340.00, date: "2026-09-21", description: "Dinner – La Maison", category: "Dining", is_recurring: false },
    { id: "t8", amount: -200.00, date: "2026-09-20", description: "Freelance Invoice #12", category: "Income", is_recurring: false },
  ],
  pending_actions: [],
};

const MOCK_TRANSACTIONS = MOCK_DASHBOARD.recent_transactions;

const MOCK_CHART_DATA = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(Date.now() + i * 86400000).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
  "Current Trajectory": 24750 + (i * 85) + Math.sin(i * 0.4) * 300,
}));
