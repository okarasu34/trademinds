import { useState, useEffect, useCallback } from "react";
import { dashboardApi, botApi, tradesApi, calendarApi } from "../utils/api";
import { useAuthStore, useBotStore, useTradesStore } from "../store";
import { useWebSocket } from "../hooks/useWebSocket";
import toast from "react-hot-toast";

const C = {
  bg: "#080c14", surface: "#0d1421", card: "#111827",
  border: "#1e2d45", accent: "#3b82f6", green: "#10b981",
  red: "#ef4444", yellow: "#f59e0b", purple: "#8b5cf6",
  text: "#f1f5f9", muted: "#64748b", dim: "#94a3b8",
};

const Badge = ({ children, color = "gray" }: { children: any; color?: string }) => {
  const map: any = {
    green: { bg: "#064e3b", text: "#34d399" }, red: { bg: "#7f1d1d", text: "#f87171" },
    blue: { bg: "#1e3a5f", text: "#60a5fa" }, yellow: { bg: "#78350f", text: "#fbbf24" },
    gray: { bg: "#1e293b", text: "#94a3b8" }, purple: { bg: "#3b0764", text: "#c084fc" },
  };
  const c = map[color] || map.gray;
  return <span style={{ background: c.bg, color: c.text, padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 600 }}>{children}</span>;
};

const MiniChart = ({ data, color, h = 36 }: { data: number[]; color: string; h?: number }) => {
  if (!data.length) return null;
  const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
  const W = 80;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * W},${h - ((v - min) / range) * h}`).join(" ");
  return (
    <svg width={W} height={h} viewBox={`0 0 ${W} ${h}`}>
      <defs><linearGradient id={`g${color.replace("#","")}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity=".3"/><stop offset="100%" stopColor={color} stopOpacity="0"/></linearGradient></defs>
      <polygon points={`0,${h} ${pts} ${W},${h}`} fill={`url(#g${color.replace("#","")})`}/>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
};

const EquityChart = ({ data }: { data: number[] }) => {
  if (!data.length) return <div style={{ height: 60, display: "flex", alignItems: "center", justifyContent: "center", color: C.muted, fontSize: 12 }}>No data yet</div>;
  const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * 100},${60 - ((v - min) / range) * 60}`).join(" ");
  return (
    <svg width="100%" height={60} viewBox="0 0 100 60" preserveAspectRatio="none">
      <defs><linearGradient id="eqg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#3b82f6" stopOpacity=".4"/><stop offset="100%" stopColor="#3b82f6" stopOpacity="0"/></linearGradient></defs>
      <polygon points={`0,60 ${pts} 100,60`} fill="url(#eqg)"/>
      <polyline points={pts} fill="none" stroke="#3b82f6" strokeWidth="0.8" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
};

const TABS = [
  { id: "dashboard", label: "Dashboard", icon: "⬛" },
  { id: "positions", label: "Positions", icon: "📊" },
  { id: "history", label: "History", icon: "📋" },
  { id: "strategies", label: "Strategies", icon: "🧠" },
  { id: "calendar", label: "Calendar", icon: "📅" },
  { id: "settings", label: "Settings", icon: "⚙️" },
];

export default function Dashboard() {
  const { user } = useAuthStore();
  const { status: botStatus, tradeMode, openPositions, updateFromApi, setBotStatus } = useBotStore();
  const { openTrades, setOpenTrades } = useTradesStore();

  const [tab, setTab] = useState("dashboard");
  const [summary, setSummary] = useState<any>(null);
  const [equityCurve, setEquityCurve] = useState<number[]>([]);
  const [calendar, setCalendar] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [selectedTrade, setSelectedTrade] = useState<any>(null);
  const [confirmClose, setConfirmClose] = useState<any>(null);
  const [closing, setClosing] = useState(false);
  const [botLoading, setBotLoading] = useState(false);
  const [time, setTime] = useState(new Date());

  // Tick clock
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // Load dashboard data
  const loadSummary = useCallback(async () => {
    try {
      const res = await dashboardApi.getSummary();
      setSummary(res.data);
      updateFromApi({
        status: res.data.bot.status,
        tradeMode: res.data.bot.trade_mode,
        openPositions: res.data.positions.open_count,
      });
      setOpenTrades(res.data.positions.trades || []);
    } catch (e) {
      console.error("Dashboard load error", e);
    }
  }, []);

  const loadEquityCurve = useCallback(async () => {
    try {
      const res = await dashboardApi.getSummary();
      // Use cumulative from equity endpoint
      const eq = await fetch("/api/v1/dashboard/equity-curve?days=30", {
        headers: { Authorization: `Bearer ${localStorage.getItem("access_token")}` }
      }).then(r => r.json());
      setEquityCurve(eq.map((p: any) => p.cumulative));
    } catch (e) {}
  }, []);

  const loadCalendar = useCallback(async () => {
    try {
      const res = await calendarApi.get({ hours_ahead: 24, impact: "high,medium" });
      setCalendar(res.data);
    } catch (e) {}
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const res = await tradesApi.list({ page_size: 50 });
      setHistory(res.data.trades || []);
    } catch (e) {}
  }, []);

  useEffect(() => {
    loadSummary();
    loadEquityCurve();
    loadCalendar();
  }, []);

  useEffect(() => {
    if (tab === "history") loadHistory();
  }, [tab]);

  // Refresh every 30s
  useEffect(() => {
    const t = setInterval(loadSummary, 30000);
    return () => clearInterval(t);
  }, []);

  // WebSocket
  const handleWsMessage = useCallback((channel: string, data: any) => {
    if (channel?.includes("trades")) {
      if (data.event === "trade_opened") toast.success(`🟢 ${data.symbol} ${data.side?.toUpperCase()} opened`);
      if (data.event === "trade_closed") toast(data.pnl >= 0 ? `💰 ${data.symbol} +${data.pnl?.toFixed(2)}` : `📉 ${data.symbol} ${data.pnl?.toFixed(2)}`);
      loadSummary();
    }
    if (channel?.includes("health")) {
      updateFromApi({ status: data.status, openPositions: data.open_positions });
    }
  }, [loadSummary]);

  useWebSocket(user?.id || null, handleWsMessage);

  // Bot controls
  const handleBot = async (action: "start" | "stop" | "pause") => {
    setBotLoading(true);
    try {
      if (action === "start") await botApi.start();
      else if (action === "stop") await botApi.stop();
      else await botApi.pause();
      setBotStatus(action === "start" ? "running" : action === "stop" ? "stopped" : "paused");
      toast.success(`Bot ${action}ed`);
      await loadSummary();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Action failed");
    } finally {
      setBotLoading(false);
    }
  };

  // Manual close
  const handleManualClose = async (tradeId: string) => {
    setClosing(true);
    try {
      await tradesApi.manualClose(tradeId);
      toast.success("Position closed manually");
      setConfirmClose(null);
      setSelectedTrade(null);
      await loadSummary();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Close failed");
    } finally {
      setClosing(false);
    }
  };

  const s = summary;
  const botColor = botStatus === "running" ? C.green : botStatus === "paused" ? C.yellow : C.red;

  return (
    <div style={{ background: C.bg, minHeight: "100vh", fontFamily: "'IBM Plex Mono', monospace", color: C.text, display: "flex" }}>

      {/* Confirm Close Modal */}
      {confirmClose && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.8)", zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 32, width: 380 }}>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>Close Position</div>
            <div style={{ color: C.dim, fontSize: 13, marginBottom: 20 }}>
              Manually close <strong>{confirmClose.symbol}</strong>? Current P&L:{" "}
              <span style={{ color: (confirmClose.pnl || 0) >= 0 ? C.green : C.red, fontWeight: 700 }}>
                {(confirmClose.pnl || 0) >= 0 ? "+" : ""}{(confirmClose.pnl || 0).toFixed(2)}
              </span>
            </div>
            <div style={{ display: "flex", gap: 10 }}>
              <button onClick={() => handleManualClose(confirmClose.id)} disabled={closing}
                style={{ flex: 1, background: C.red, color: "white", border: "none", borderRadius: 8, padding: "10px 0", fontWeight: 700, cursor: "pointer", fontFamily: "inherit", opacity: closing ? 0.7 : 1 }}>
                {closing ? "Closing..." : "Close Now"}
              </button>
              <button onClick={() => setConfirmClose(null)}
                style={{ flex: 1, background: C.surface, color: C.dim, border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 0", cursor: "pointer", fontFamily: "inherit" }}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar */}
      <div style={{ width: 200, background: C.surface, borderRight: `1px solid ${C.border}`, display: "flex", flexDirection: "column", position: "fixed", top: 0, bottom: 0, zIndex: 50 }}>
        <div style={{ padding: "22px 18px 16px", borderBottom: `1px solid ${C.border}` }}>
          <div style={{ fontSize: 15, fontWeight: 800 }}><span style={{ color: C.accent }}>Trade</span>Minds</div>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>AI TRADING ENGINE</div>
        </div>

        {/* Bot status */}
        <div style={{ padding: "12px 14px", borderBottom: `1px solid ${C.border}` }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: botColor, boxShadow: botStatus === "running" ? `0 0 6px ${C.green}` : "none" }} />
            <span style={{ fontSize: 11, fontWeight: 600, color: C.dim, textTransform: "uppercase" }}>{botStatus}</span>
          </div>
          <div style={{ display: "flex", gap: 5 }}>
            {(["start", "pause", "stop"] as const).map(a => (
              <button key={a} onClick={() => handleBot(a)} disabled={botLoading}
                style={{ flex: 1, fontSize: 9, padding: "4px 0", background: (a === "start" && botStatus === "running") || (a === "pause" && botStatus === "paused") || (a === "stop" && botStatus === "stopped") ? C.accent : C.card, color: "white", border: "none", borderRadius: 4, cursor: "pointer", fontFamily: "inherit", fontWeight: 700, opacity: botLoading ? 0.6 : 1 }}>
                {a === "start" ? "▶" : a === "pause" ? "⏸" : "⏹"}
              </button>
            ))}
          </div>
          <div style={{ marginTop: 8, fontSize: 10, color: C.muted }}>
            Mode: <span style={{ color: tradeMode === "live" ? C.green : C.yellow, fontWeight: 700, textTransform: "uppercase" }}>{tradeMode}</span>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: "8px 8px" }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              style={{ display: "flex", alignItems: "center", gap: 10, width: "100%", padding: "10px 12px", marginBottom: 2, background: tab === t.id ? C.card : "transparent", border: tab === t.id ? `1px solid ${C.border}` : "1px solid transparent", borderRadius: 8, color: tab === t.id ? C.text : C.muted, cursor: "pointer", fontFamily: "inherit", fontSize: 12, fontWeight: tab === t.id ? 700 : 400, textAlign: "left" }}>
              <span style={{ fontSize: 13 }}>{t.icon}</span> {t.label}
              {t.id === "positions" && openPositions > 0 && (
                <span style={{ marginLeft: "auto", background: C.accent, color: "white", borderRadius: "50%", width: 18, height: 18, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700 }}>{openPositions}</span>
              )}
            </button>
          ))}
        </nav>

        <div style={{ padding: "10px 14px", borderTop: `1px solid ${C.border}`, fontSize: 11, color: C.muted }}>
          <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.08em" }}>{time.toTimeString().slice(0, 8)}</div>
          <div style={{ fontSize: 10, marginTop: 2 }}>UTC</div>
        </div>
      </div>

      {/* Main */}
      <div style={{ marginLeft: 200, flex: 1, minHeight: "100vh" }}>

        {/* Topbar */}
        <div style={{ position: "sticky", top: 0, zIndex: 40, background: `${C.surface}ee`, backdropFilter: "blur(12px)", borderBottom: `1px solid ${C.border}`, padding: "0 24px", height: 50, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em" }}>
            {TABS.find(t => t.id === tab)?.label}
          </div>
          <div style={{ fontSize: 11, color: C.muted }}>
            {user?.email} · {user?.base_currency}
          </div>
        </div>

        {/* ── Dashboard Tab ── */}
        {tab === "dashboard" && s && (
          <div style={{ padding: 24 }}>
            {/* KPIs */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 14, marginBottom: 20 }}>
              {[
                { label: "Open Positions", value: `${s.positions.open_count} / ${s.bot.max_positions}`, color: C.accent },
                { label: "Unrealized P&L", value: `${s.positions.unrealized_pnl >= 0 ? "+" : ""}${s.positions.unrealized_pnl.toFixed(2)}`, color: s.positions.unrealized_pnl >= 0 ? C.green : C.red },
                { label: "Today P&L", value: `${s.today.pnl >= 0 ? "+" : ""}${s.today.pnl.toFixed(2)}`, color: s.today.pnl >= 0 ? C.green : C.red },
                { label: "All-Time P&L", value: `${s.all_time.pnl >= 0 ? "+" : ""}${s.all_time.pnl.toFixed(2)}`, color: s.all_time.pnl >= 0 ? C.green : C.red },
                { label: "Win Rate", value: `${s.all_time.win_rate.toFixed(1)}%`, color: s.all_time.win_rate >= 50 ? C.green : C.red },
              ].map((k, i) => (
                <div key={i} style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 16 }}>
                  <div style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>{k.label}</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: k.color }}>{k.value}</div>
                </div>
              ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 16 }}>
              {/* Equity Curve */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 18 }}>
                <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Equity Curve (30 days)</div>
                <EquityChart data={equityCurve} />
              </div>

              {/* AI Signals */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 18 }}>
                <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Latest AI Signals</div>
                {(s.signals || []).slice(0, 6).map((sig: any, i: number) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "7px 10px", background: C.surface, borderRadius: 6, marginBottom: 6 }}>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span style={{ fontSize: 12, fontWeight: 700 }}>{sig.symbol}</span>
                      <Badge color={sig.signal === "buy" ? "green" : sig.signal === "sell" ? "red" : "gray"}>{sig.signal.toUpperCase()}</Badge>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <div style={{ width: 36, height: 3, background: C.border, borderRadius: 2 }}>
                        <div style={{ width: `${sig.confidence * 100}%`, height: "100%", background: sig.confidence > 0.75 ? C.green : C.yellow, borderRadius: 2 }} />
                      </div>
                      <span style={{ fontSize: 10, color: C.muted }}>{(sig.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
                {!s.signals?.length && <div style={{ color: C.muted, fontSize: 12, textAlign: "center", padding: 16 }}>No signals yet</div>}
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {/* Open Positions */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 18 }}>
                <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Open Positions</div>
                {(s.positions.trades || []).map((t: any) => (
                  <div key={t.id} style={{ padding: "10px 12px", background: C.surface, borderRadius: 8, marginBottom: 8, cursor: "pointer" }} onClick={() => setSelectedTrade(selectedTrade?.id === t.id ? null : t)}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                        <span style={{ fontWeight: 700, fontSize: 12 }}>{t.symbol}</span>
                        <Badge color={t.side === "buy" ? "green" : "red"}>{t.side.toUpperCase()}</Badge>
                      </div>
                      <span style={{ fontWeight: 700, fontSize: 12, color: (t.pnl || 0) >= 0 ? C.green : C.red }}>
                        {(t.pnl || 0) >= 0 ? "+" : ""}{(t.pnl || 0).toFixed(2)}
                      </span>
                    </div>
                    <div style={{ fontSize: 10, color: C.muted, marginTop: 4 }}>Entry: {t.entry_price} · Lot: {t.lot_size}</div>
                    {selectedTrade?.id === t.id && (
                      <div style={{ marginTop: 10, paddingTop: 10, borderTop: `1px solid ${C.border}` }}>
                        <button onClick={e => { e.stopPropagation(); setConfirmClose(t); }}
                          style={{ width: "100%", background: C.red, color: "white", border: "none", borderRadius: 6, padding: "8px 0", fontSize: 11, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>
                          CLOSE MANUALLY
                        </button>
                      </div>
                    )}
                  </div>
                ))}
                {!s.positions.trades?.length && <div style={{ color: C.muted, fontSize: 12, textAlign: "center", padding: 20 }}>No open positions</div>}
              </div>

              {/* Economic Calendar */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 18 }}>
                <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>📅 Economic Calendar</div>
                {(s.calendar || []).map((e: any, i: number) => (
                  <div key={i} style={{ padding: "9px 12px", background: C.surface, borderRadius: 8, marginBottom: 7, borderLeft: `3px solid ${e.impact === "high" ? C.red : e.impact === "medium" ? C.yellow : C.muted}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                      <div style={{ fontSize: 11, fontWeight: 600, flex: 1 }}>{e.title}</div>
                      <span style={{ fontSize: 10, color: C.muted, marginLeft: 8, whiteSpace: "nowrap" }}>{e.minutes_until > 0 ? `in ${e.minutes_until}m` : "now"}</span>
                    </div>
                    <div style={{ display: "flex", gap: 10, marginTop: 4, fontSize: 10, color: C.muted }}>
                      <span>{e.currency}</span>
                      <Badge color={e.impact === "high" ? "red" : e.impact === "medium" ? "yellow" : "gray"}>{e.impact}</Badge>
                      {e.forecast && <span>F: {e.forecast}</span>}
                      {e.previous && <span>P: {e.previous}</span>}
                      {e.actual && <span style={{ color: C.green }}>A: {e.actual}</span>}
                    </div>
                  </div>
                ))}
                {!s.calendar?.length && <div style={{ color: C.muted, fontSize: 12, textAlign: "center", padding: 20 }}>No upcoming events</div>}
              </div>
            </div>
          </div>
        )}

        {/* ── Positions Tab ── */}
        {tab === "positions" && (
          <div style={{ padding: 24 }}>
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, overflow: "hidden" }}>
              <div style={{ padding: "14px 20px", borderBottom: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontSize: 13, fontWeight: 700 }}>Open Positions ({openPositions})</div>
                <div style={{ fontSize: 11, color: C.muted }}>Open decisions are fully autonomous</div>
              </div>
              {openTrades.length === 0 ? (
                <div style={{ padding: 40, textAlign: "center", color: C.muted }}>No open positions</div>
              ) : (
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ background: C.surface }}>
                      {["Symbol", "Side", "Entry", "Lot", "SL", "TP", "P&L", "Confidence", "Opened", "Action"].map(h => (
                        <th key={h} style={{ padding: "10px 14px", fontSize: 10, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.06em", textAlign: "left", borderBottom: `1px solid ${C.border}` }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {openTrades.map((t: any) => (
                      <tr key={t.id} style={{ borderBottom: `1px solid ${C.border}` }}>
                        <td style={{ padding: "12px 14px", fontWeight: 700, fontSize: 13 }}>{t.symbol}</td>
                        <td style={{ padding: "12px 14px" }}><Badge color={t.side === "buy" ? "green" : "red"}>{t.side.toUpperCase()}</Badge></td>
                        <td style={{ padding: "12px 14px", fontSize: 12 }}>{t.entry_price}</td>
                        <td style={{ padding: "12px 14px", fontSize: 12 }}>{t.lot_size}</td>
                        <td style={{ padding: "12px 14px", fontSize: 12, color: C.red }}>{t.stop_loss || "—"}</td>
                        <td style={{ padding: "12px 14px", fontSize: 12, color: C.green }}>{t.take_profit || "—"}</td>
                        <td style={{ padding: "12px 14px", fontWeight: 700, fontSize: 13, color: (t.pnl || 0) >= 0 ? C.green : C.red }}>
                          {(t.pnl || 0) >= 0 ? "+" : ""}{(t.pnl || 0).toFixed(2)}
                        </td>
                        <td style={{ padding: "12px 14px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <div style={{ width: 40, height: 3, background: C.border, borderRadius: 2 }}>
                              <div style={{ width: `${(t.ai_confidence || 0) * 100}%`, height: "100%", background: (t.ai_confidence || 0) > 0.75 ? C.green : C.yellow, borderRadius: 2 }} />
                            </div>
                            <span style={{ fontSize: 10, color: C.muted }}>{((t.ai_confidence || 0) * 100).toFixed(0)}%</span>
                          </div>
                        </td>
                        <td style={{ padding: "12px 14px", fontSize: 11, color: C.muted }}>{t.opened_at?.slice(0, 16)}</td>
                        <td style={{ padding: "12px 14px" }}>
                          <button onClick={() => setConfirmClose(t)}
                            style={{ background: C.red, color: "white", border: "none", borderRadius: 6, padding: "6px 12px", fontSize: 11, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>
                            Close
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}

        {/* ── History Tab ── */}
        {tab === "history" && (
          <div style={{ padding: 24 }}>
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, overflow: "hidden" }}>
              <div style={{ padding: "14px 20px", borderBottom: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between" }}>
                <div style={{ fontSize: 13, fontWeight: 700 }}>Trade History</div>
                <div style={{ display: "flex", gap: 8 }}>
                  {["week", "month", "year", "all"].map(p => (
                    <a key={p} href={`/api/v1/reports/pdf?period=${p}`} target="_blank"
                      style={{ background: C.surface, color: C.dim, border: `1px solid ${C.border}`, borderRadius: 6, padding: "5px 12px", fontSize: 11, textDecoration: "none" }}>
                      PDF ({p})
                    </a>
                  ))}
                </div>
              </div>
              {history.length === 0 ? (
                <div style={{ padding: 40, textAlign: "center", color: C.muted }}>No trade history yet</div>
              ) : (
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ background: C.surface }}>
                      {["Symbol", "Market", "Side", "Entry", "Exit", "Lot", "P&L", "Strategy", "Closed By", "Date"].map(h => (
                        <th key={h} style={{ padding: "10px 14px", fontSize: 10, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.06em", textAlign: "left", borderBottom: `1px solid ${C.border}` }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((t: any) => (
                      <tr key={t.id} style={{ borderBottom: `1px solid ${C.border}`, cursor: "pointer" }} onClick={() => setSelectedTrade(selectedTrade?.id === t.id ? null : t)}>
                        <td style={{ padding: "11px 14px", fontWeight: 700, fontSize: 13 }}>{t.symbol}</td>
                        <td style={{ padding: "11px 14px" }}><Badge color="gray">{t.market_type}</Badge></td>
                        <td style={{ padding: "11px 14px" }}><Badge color={t.side === "buy" ? "green" : "red"}>{t.side.toUpperCase()}</Badge></td>
                        <td style={{ padding: "11px 14px", fontSize: 12 }}>{t.entry_price}</td>
                        <td style={{ padding: "11px 14px", fontSize: 12 }}>{t.exit_price || "—"}</td>
                        <td style={{ padding: "11px 14px", fontSize: 12 }}>{t.lot_size}</td>
                        <td style={{ padding: "11px 14px", fontWeight: 700, fontSize: 13, color: (t.pnl || 0) >= 0 ? C.green : C.red }}>
                          {(t.pnl || 0) >= 0 ? "+" : ""}{(t.pnl || 0).toFixed(2)}
                        </td>
                        <td style={{ padding: "11px 14px", fontSize: 11, color: C.dim }}>{t.strategy_name || "—"}</td>
                        <td style={{ padding: "11px 14px" }}><Badge color={t.closed_by === "manual" ? "yellow" : "gray"}>{t.closed_by || "bot"}</Badge></td>
                        <td style={{ padding: "11px 14px", fontSize: 11, color: C.muted }}>{t.opened_at?.slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {selectedTrade?.ai_reasoning && (
                <div style={{ padding: 16, borderTop: `1px solid ${C.border}`, background: C.surface }}>
                  <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 6 }}>AI Reasoning — {selectedTrade.symbol}</div>
                  <div style={{ fontSize: 11, color: C.dim, lineHeight: 1.7 }}>{selectedTrade.ai_reasoning}</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Calendar Tab ── */}
        {tab === "calendar" && (
          <div style={{ padding: 24 }}>
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, overflow: "hidden" }}>
              <div style={{ padding: "14px 20px", borderBottom: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontSize: 13, fontWeight: 700 }}>Economic Calendar — MyFXBook</div>
                <Badge color="green">Live XML Feed</Badge>
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ background: C.surface }}>
                    {["Time", "Event", "Currency", "Impact", "Forecast", "Previous", "Actual"].map(h => (
                      <th key={h} style={{ padding: "10px 16px", fontSize: 10, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.06em", textAlign: "left", borderBottom: `1px solid ${C.border}` }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {calendar.map((e: any, i: number) => (
                    <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                      <td style={{ padding: "13px 16px", fontWeight: 700, fontSize: 12 }}>
                        {e.minutes_until > 0 ? `in ${e.minutes_until}m` : <span style={{ color: C.green }}>NOW</span>}
                      </td>
                      <td style={{ padding: "13px 16px", fontSize: 12 }}>{e.title}</td>
                      <td style={{ padding: "13px 16px" }}><Badge color="gray">{e.currency}</Badge></td>
                      <td style={{ padding: "13px 16px" }}><Badge color={e.impact === "high" ? "red" : e.impact === "medium" ? "yellow" : "gray"}>{e.impact.toUpperCase()}</Badge></td>
                      <td style={{ padding: "13px 16px", fontSize: 12 }}>{e.forecast || "—"}</td>
                      <td style={{ padding: "13px 16px", fontSize: 12 }}>{e.previous || "—"}</td>
                      <td style={{ padding: "13px 16px", fontSize: 12, fontWeight: e.actual ? 700 : 400, color: e.actual ? C.green : C.muted }}>{e.actual || "—"}</td>
                    </tr>
                  ))}
                  {!calendar.length && (
                    <tr><td colSpan={7} style={{ padding: 40, textAlign: "center", color: C.muted }}>No upcoming events</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── Settings Tab ── */}
        {tab === "settings" && <SettingsPanel />}

        {/* ── Strategies Tab ── */}
        {tab === "strategies" && <StrategiesPanel />}

        {/* Loading state */}
        {!s && tab === "dashboard" && (
          <div style={{ padding: 40, textAlign: "center", color: C.muted }}>Loading...</div>
        )}
      </div>
    </div>
  );
}

// ── Settings Panel ──
function SettingsPanel() {
  const [config, setConfig] = useState<any>({});
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    botApi.getStatus().then(r => setConfig(r.data)).catch(() => {});
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      await botApi.updateConfig({
        max_positions: config.max_positions,
        max_daily_loss_pct: config.max_daily_loss_pct,
        max_risk_per_trade_pct: config.max_risk_per_trade_pct,
        news_pause_minutes: config.news_pause_minutes,
        pause_on_high_impact_news: config.pause_on_high_impact_news,
        market_limits: config.market_limits,
      });
      toast.success("Settings saved");
    } catch {
      toast.error("Save failed");
    } finally {
      setSaving(false);
    }
  };

  const fields = [
    { key: "max_positions", label: "Max Open Positions", min: 1, max: 100, step: 1, unit: "positions" },
    { key: "max_daily_loss_pct", label: "Max Daily Loss", min: 0.5, max: 20, step: 0.5, unit: "% of balance" },
    { key: "max_risk_per_trade_pct", label: "Max Risk Per Trade", min: 0.1, max: 5, step: 0.1, unit: "% of balance" },
    { key: "news_pause_minutes", label: "News Pause Window", min: 5, max: 120, step: 5, unit: "minutes before" },
  ];

  return (
    <div style={{ padding: 24, maxWidth: 680 }}>
      <div style={{ background: "#111827", border: `1px solid #1e2d45`, borderRadius: 10, padding: 24, marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 20 }}>Risk Limits</div>
        {fields.map(f => (
          <div key={f.key} style={{ marginBottom: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
              <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>{f.label}</label>
              <span style={{ fontSize: 12, fontWeight: 700, color: "#3b82f6" }}>{config[f.key]} {f.unit}</span>
            </div>
            <input type="range" min={f.min} max={f.max} step={f.step} value={config[f.key] || f.min}
              onChange={e => setConfig((prev: any) => ({ ...prev, [f.key]: Number(e.target.value) }))}
              style={{ width: "100%", accentColor: "#3b82f6" }} />
          </div>
        ))}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: 14, borderTop: `1px solid #1e2d45` }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Pause on High-Impact News</div>
            <div style={{ fontSize: 11, color: "#64748b" }}>Bot stops before major events</div>
          </div>
          <button onClick={() => setConfig((p: any) => ({ ...p, pause_on_high_impact_news: !p.pause_on_high_impact_news }))}
            style={{ width: 44, height: 24, borderRadius: 12, border: "none", background: config.pause_on_high_impact_news ? "#10b981" : "#1e2d45", cursor: "pointer", position: "relative", transition: "background 0.2s" }}>
            <div style={{ width: 18, height: 18, borderRadius: "50%", background: "white", position: "absolute", top: 3, left: config.pause_on_high_impact_news ? 23 : 3, transition: "left 0.2s" }} />
          </button>
        </div>
      </div>

      <div style={{ background: "#111827", border: `1px solid #1e2d45`, borderRadius: 10, padding: 24, marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 16 }}>Trade Mode</div>
        <div style={{ display: "flex", gap: 10 }}>
          {["paper", "live"].map(m => (
            <button key={m} onClick={async () => { await botApi.setMode(m); setConfig((p: any) => ({ ...p, trade_mode: m })); toast.success(`Mode: ${m}`); }}
              style={{ flex: 1, padding: "12px 0", background: config.trade_mode === m ? (m === "live" ? "#10b981" : "#3b82f6") : "#0d1421", color: "white", border: `1px solid #1e2d45`, borderRadius: 8, fontFamily: "inherit", fontWeight: 700, fontSize: 12, textTransform: "uppercase", cursor: "pointer" }}>
              {m}
            </button>
          ))}
        </div>
        {config.trade_mode === "live" && (
          <div style={{ marginTop: 12, background: "#78350f", border: "1px solid #92400e", borderRadius: 8, padding: "10px 14px", fontSize: 12, color: "#fbbf24" }}>
            ⚠️ Live mode uses real funds. Ensure thorough paper testing before proceeding.
          </div>
        )}
      </div>

      <button onClick={save} disabled={saving}
        style={{ width: "100%", background: "#3b82f6", color: "white", border: "none", borderRadius: 8, padding: "14px 0", fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: "inherit", opacity: saving ? 0.7 : 1 }}>
        {saving ? "Saving..." : "Save Settings"}
      </button>
    </div>
  );
}

// ── Strategies Panel ──
function StrategiesPanel() {
  const [strategies, setStrategies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    import("../utils/api").then(({ strategiesApi }) => {
      strategiesApi.list().then(r => { setStrategies(r.data); setLoading(false); }).catch(() => setLoading(false));
    });
  }, []);

  const toggle = async (id: string) => {
    const { strategiesApi } = await import("../utils/api");
    const r = await strategiesApi.toggle(id);
    setStrategies(prev => prev.map(s => s.id === id ? { ...s, is_active: r.data.is_active } : s));
  };

  if (loading) return <div style={{ padding: 40, textAlign: "center", color: "#64748b" }}>Loading...</div>;

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
        {strategies.map(s => (
          <div key={s.id} style={{ background: "#111827", border: `1px solid #1e2d45`, borderRadius: 10, padding: 18 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
              <div style={{ fontSize: 13, fontWeight: 700 }}>{s.name}</div>
              <button onClick={() => toggle(s.id)}
                style={{ width: 40, height: 22, borderRadius: 11, border: "none", background: s.is_active ? "#10b981" : "#1e2d45", cursor: "pointer", position: "relative", flexShrink: 0 }}>
                <div style={{ width: 16, height: 16, borderRadius: "50%", background: "white", position: "absolute", top: 3, left: s.is_active ? 21 : 3, transition: "left 0.2s" }} />
              </button>
            </div>
            <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 12, lineHeight: 1.6 }}>{s.description}</div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
              {(s.markets || []).map((m: string) => (
                <Badge key={m} color="gray">{m}</Badge>
              ))}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <div style={{ background: "#0d1421", borderRadius: 6, padding: "8px 10px" }}>
                <div style={{ fontSize: 10, color: "#64748b" }}>Win Rate</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: s.win_rate >= 50 ? "#10b981" : "#f59e0b" }}>{s.win_rate?.toFixed(1) || 0}%</div>
              </div>
              <div style={{ background: "#0d1421", borderRadius: 6, padding: "8px 10px" }}>
                <div style={{ fontSize: 10, color: "#64748b" }}>Total P&L</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: (s.total_pnl || 0) >= 0 ? "#10b981" : "#ef4444" }}>
                  {(s.total_pnl || 0) >= 0 ? "+" : ""}{(s.total_pnl || 0).toFixed(2)}
                </div>
              </div>
            </div>
            <div style={{ marginTop: 10, fontSize: 10, color: "#64748b" }}>{s.total_trades} trades</div>
          </div>
        ))}
      </div>
    </div>
  );
}
