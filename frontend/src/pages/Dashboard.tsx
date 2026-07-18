import { useState, useEffect, useCallback } from "react";
import { dashboardApi, botApi, tradesApi, calendarApi } from "../utils/api";
import { useAuthStore, useBotStore, useTradesStore } from "../store";
import { useWebSocket } from "../hooks/useWebSocket";
import toast from "react-hot-toast";

const C = {
  bg: "#0f1f33", surface: "#162840", card: "#1a3248",
  border: "#1e3a5f", accent: "#2dd4bf", green: "#2dd4bf",
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
  { id: "brokers", label: "Brokers", icon: "🔗" },
  { id: "backtest", label: "Backtest", icon: "🧪" },
  { id: "performance", label: "Performance", icon: "📈" },
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
                { label: "Balance", value: `${s.account.balance.toFixed(2)}`, color: C.green },
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
        {tab === "brokers" && <BrokersPanel />}

        {/* ── Backtest Tab ── */}
        {tab === "backtest" && <BacktestPanel />}

        {/* ── Performance Tab ── */}
        {tab === "performance" && <PerformancePanel />}

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
   setStrategies(prev => prev.map(s => s.id === id ? { ...s, is_active: true } : { ...s, is_active: false }));
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

// ── Brokers Panel ──
function BrokersPanel() {
  const [brokers, setBrokers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [testing, setTesting] = useState<string | null>(null);
  const [form, setForm] = useState({ name: "", broker_type: "capital", api_key: "", api_secret: "", extra: "", market_type: "MULTI" });
  const [editForm, setEditForm] = useState({ name: "", api_key: "", api_secret: "", extra: "" });

  useEffect(() => {
    loadBrokers();
  }, []);

  const loadBrokers = async () => {
    try {
      const { brokersApi } = await import("../utils/api");
      const r = await brokersApi.list();
      setBrokers(r.data);
    } catch {}
    setLoading(false);
  };

  const toggle = async (id: string) => {
    const { brokersApi } = await import("../utils/api");
    await brokersApi.toggle(id);
    await loadBrokers();
  };

  const testBroker = async (id: string) => {
    setTesting(id);
    try {
      const { brokersApi } = await import("../utils/api");
      const r = await brokersApi.test(id);
      if (r.data.connected) {
        toast.success(`Connected! Balance: ${r.data.balance} ${r.data.currency}`);
      } else {
        toast.error(`Connection failed: ${r.data.error}`);
      }
    } catch (e: any) {
      toast.error("Test failed");
    }
    setTesting(null);
    await loadBrokers();
  };

  const addBroker = async () => {
    try {
      const { brokersApi } = await import("../utils/api");
      await brokersApi.add(form);
      toast.success("Broker added!");
      setShowAdd(false);
      setForm({ name: "", broker_type: "capital", api_key: "", api_secret: "", extra: "", market_type: "MULTI" });
      await loadBrokers();
    } catch (e: any) {
      toast.error("Failed to add broker");
    }
  };

  const startEdit = (b: any) => {
    setEditing(b.id);
    setEditForm({ name: b.name, api_key: "", api_secret: "", extra: "" });
  };

  const saveEdit = async (id: string) => {
    try {
      const { brokersApi } = await import("../utils/api");
      const payload: any = {};
      if (editForm.name) payload.name = editForm.name;
      if (editForm.api_key) payload.api_key = editForm.api_key;
      if (editForm.api_secret) payload.api_secret = editForm.api_secret;
      if (editForm.extra) payload.extra = editForm.extra;
      await brokersApi.update(id, payload);
      toast.success("Broker updated! Test connection to verify.");
      setEditing(null);
      await loadBrokers();
    } catch {
      toast.error("Failed to update");
    }
  };

  const removeBroker = async (id: string) => {
    if (!confirm("Are you sure?")) return;
    try {
      const { brokersApi } = await import("../utils/api");
      await brokersApi.remove(id);
      toast.success("Broker removed");
      await loadBrokers();
    } catch {
      toast.error("Failed to remove");
    }
  };

  if (loading) return <div style={{ padding: 40, textAlign: "center", color: "#64748b" }}>Loading...</div>;

  const brokerTypes: Record<string, string> = {
    capital: "Capital.com", ig: "IG Markets", ibkr: "Interactive Brokers",
    mt5: "MetaTrader 5",
};

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
        <div style={{ fontSize: 15, fontWeight: 700 }}>Broker Accounts</div>
        <button onClick={() => setShowAdd(!showAdd)}
          style={{ background: "#3b82f6", color: "white", border: "none", borderRadius: 6, padding: "8px 16px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>
          {showAdd ? "Cancel" : "+ Add Broker"}
        </button>
      </div>

      {showAdd && (
        <div style={{ background: "#111827", border: "1px solid #1e2d45", borderRadius: 10, padding: 20, marginBottom: 20 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <label style={{ fontSize: 11, color: "#64748b", display: "block", marginBottom: 4 }}>Name</label>
              <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })}
                placeholder="e.g. IG Demo"
                style={{ width: "100%", background: "#0d1421", border: "1px solid #1e2d45", borderRadius: 6, padding: "8px 10px", color: "#f1f5f9", fontSize: 13, outline: "none", boxSizing: "border-box" }} />
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#64748b", display: "block", marginBottom: 4 }}>Broker Type</label>
              <select value={form.broker_type} onChange={e => setForm({ ...form, broker_type: e.target.value })}
                style={{ width: "100%", background: "#0d1421", border: "1px solid #1e2d45", borderRadius: 6, padding: "8px 10px", color: "#f1f5f9", fontSize: 13, outline: "none", boxSizing: "border-box" }}>
                {Object.entries(brokerTypes).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#64748b", display: "block", marginBottom: 4 }}>API Key</label>
              <input value={form.api_key} onChange={e => setForm({ ...form, api_key: e.target.value })}
                style={{ width: "100%", background: "#0d1421", border: "1px solid #1e2d45", borderRadius: 6, padding: "8px 10px", color: "#f1f5f9", fontSize: 13, outline: "none", boxSizing: "border-box" }} />
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#64748b", display: "block", marginBottom: 4 }}>Password</label>
              <input type="password" value={form.api_secret} onChange={e => setForm({ ...form, api_secret: e.target.value })}
                style={{ width: "100%", background: "#0d1421", border: "1px solid #1e2d45", borderRadius: 6, padding: "8px 10px", color: "#f1f5f9", fontSize: 13, outline: "none", boxSizing: "border-box" }} />
            </div>
            <div style={{ gridColumn: "1 / -1" }}>
              <label style={{ fontSize: 11, color: "#64748b", display: "block", marginBottom: 4 }}>Email / Identifier (optional)</label>
              <input value={form.extra} onChange={e => setForm({ ...form, extra: e.target.value })}
                placeholder="e.g. your@email.com"
                style={{ width: "100%", background: "#0d1421", border: "1px solid #1e2d45", borderRadius: 6, padding: "8px 10px", color: "#f1f5f9", fontSize: 13, outline: "none", boxSizing: "border-box" }} />
            </div>
          </div>
          <button onClick={addBroker}
            style={{ marginTop: 14, background: "#10b981", color: "white", border: "none", borderRadius: 6, padding: "10px 24px", fontSize: 13, fontWeight: 600, cursor: "pointer" }}>
            Add Broker
          </button>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 14 }}>
        {brokers.map(b => (
          <div key={b.id} style={{
            background: "#111827",
            border: `1px solid ${b.is_active ? "#10b981" : "#1e2d45"}`,
            borderRadius: 10,
            padding: 18,
            opacity: b.is_active ? 1 : 0.7,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700 }}>{b.name}</div>
                <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{brokerTypes[b.broker_type] || b.broker_type}</div>
              </div>
              <button onClick={() => toggle(b.id)}
                style={{ width: 40, height: 22, borderRadius: 11, border: "none", background: b.is_active ? "#10b981" : "#1e2d45", cursor: "pointer", position: "relative", flexShrink: 0 }}>
                <div style={{ width: 16, height: 16, borderRadius: "50%", background: "white", position: "absolute", top: 3, left: b.is_active ? 21 : 3, transition: "left 0.2s" }} />
              </button>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
              <div style={{ background: "#0d1421", borderRadius: 6, padding: "8px 10px" }}>
                <div style={{ fontSize: 10, color: "#64748b" }}>Balance</div>
                <div style={{ fontSize: 14, fontWeight: 700 }}>{b.balance?.toFixed(2) || "—"} {b.currency || ""}</div>
              </div>
              <div style={{ background: "#0d1421", borderRadius: 6, padding: "8px 10px" }}>
                <div style={{ fontSize: 10, color: "#64748b" }}>Status</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: b.is_connected ? "#10b981" : "#ef4444" }}>
                  {b.is_connected ? "Connected" : "Disconnected"}
                </div>
              </div>
            </div>

            {/* Edit Form */}
            {editing === b.id && (
              <div style={{ background: "#0d1421", borderRadius: 8, padding: 14, marginBottom: 12, border: "1px solid #1e2d45" }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#60a5fa", marginBottom: 10 }}>Edit Credentials</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div>
                    <label style={{ fontSize: 10, color: "#64748b" }}>Name</label>
                    <input value={editForm.name} onChange={e => setEditForm({ ...editForm, name: e.target.value })}
                      style={{ width: "100%", background: "#111827", border: "1px solid #1e2d45", borderRadius: 4, padding: "6px 8px", color: "#f1f5f9", fontSize: 12, outline: "none", boxSizing: "border-box" }} />
                  </div>
                  <div>
                    <label style={{ fontSize: 10, color: "#64748b" }}>New API Key</label>
                    <input value={editForm.api_key} onChange={e => setEditForm({ ...editForm, api_key: e.target.value })}
                      placeholder="Leave empty to keep current"
                      style={{ width: "100%", background: "#111827", border: "1px solid #1e2d45", borderRadius: 4, padding: "6px 8px", color: "#f1f5f9", fontSize: 12, outline: "none", boxSizing: "border-box" }} />
                  </div>
                  <div>
                    <label style={{ fontSize: 10, color: "#64748b" }}>New Password</label>
                    <input type="password" value={editForm.api_secret} onChange={e => setEditForm({ ...editForm, api_secret: e.target.value })}
                      placeholder="Leave empty to keep current"
                      style={{ width: "100%", background: "#111827", border: "1px solid #1e2d45", borderRadius: 4, padding: "6px 8px", color: "#f1f5f9", fontSize: 12, outline: "none", boxSizing: "border-box" }} />
                  </div>
                  <div>
                    <label style={{ fontSize: 10, color: "#64748b" }}>New Email / Identifier</label>
                    <input value={editForm.extra} onChange={e => setEditForm({ ...editForm, extra: e.target.value })}
                      placeholder="Leave empty to keep current"
                      style={{ width: "100%", background: "#111827", border: "1px solid #1e2d45", borderRadius: 4, padding: "6px 8px", color: "#f1f5f9", fontSize: 12, outline: "none", boxSizing: "border-box" }} />
                  </div>
                  <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
                    <button onClick={() => saveEdit(b.id)}
                      style={{ flex: 1, background: "#10b981", color: "white", border: "none", borderRadius: 4, padding: "7px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>
                      Save
                    </button>
                    <button onClick={() => setEditing(null)}
                      style={{ flex: 1, background: "#1e2d45", color: "#94a3b8", border: "none", borderRadius: 4, padding: "7px", fontSize: 12, cursor: "pointer" }}>
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={() => testBroker(b.id)} disabled={testing === b.id}
                style={{ flex: 1, background: "#1e2d45", color: "#60a5fa", border: "none", borderRadius: 6, padding: "8px", fontSize: 12, fontWeight: 600, cursor: "pointer", opacity: testing === b.id ? 0.5 : 1 }}>
                {testing === b.id ? "Testing..." : "Test Connection"}
              </button>
              <button onClick={() => startEdit(b)}
                style={{ background: "#1e2d45", color: "#fbbf24", border: "none", borderRadius: 6, padding: "8px 12px", fontSize: 12, cursor: "pointer" }}>
                ✎
              </button>
              <button onClick={() => removeBroker(b.id)}
                style={{ background: "#1e2d45", color: "#f87171", border: "none", borderRadius: 6, padding: "8px 12px", fontSize: 12, cursor: "pointer" }}>
                ✕
              </button>
            </div>

            {b.last_sync && (
              <div style={{ fontSize: 10, color: "#64748b", marginTop: 8 }}>
                Last sync: {new Date(b.last_sync).toLocaleString()}
              </div>
            )}
          </div>
        ))}
      </div>

      {brokers.length === 0 && (
        <div style={{ textAlign: "center", color: "#64748b", padding: 40 }}>
          No brokers configured. Click "+ Add Broker" to get started.
        </div>
      )}
    </div>
  );
}

// ─────────────────────────── BACKTEST PANEL ───────────────────────────
function BacktestPanel() {
  const C = { bg: "#0a0f1a", card: "#0d1526", border: "#1e2d45", text: "#f1f5f9", muted: "#64748b", accent: "#3b82f6", green: "#10b981", red: "#f87171", yellow: "#fbbf24" };
  const [strategies, setStrategies] = useState<any[]>([]);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [backtests, setBacktests] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [form, setForm] = useState({ name: "", strategy_id: "", symbol: "", timeframe: "1h", start_date: "", end_date: "", initial_balance: 10000 });
  const [selected, setSelected] = useState<any>(null);
  useEffect(() => {
    import("../utils/api").then(({ strategiesApi, backtestsApi: btApi, botSymbolsApi }) => {
      strategiesApi.list().then((r: any) => setStrategies(Array.isArray(r.data) ? r.data : [])).catch(() => {});
      btApi.list().then((r: any) => setBacktests(Array.isArray(r.data) ? r.data : [])).catch(() => {});
      botSymbolsApi.list().then((r: any) => {
        const list = Array.isArray(r.data?.symbols) ? r.data.symbols : [];
        setSymbols(list);
        if (list.length > 0) setForm(f => ({ ...f, symbol: list[0] }));
      }).catch(() => {});
    });
  }, []);

  const loadBacktests = () => {
    import("../utils/api").then(({ backtestsApi: btApi }) => {
      btApi.list().then((r: any) => setBacktests(Array.isArray(r.data) ? r.data : [])).catch(() => {});
    });
  };

  const run = async () => {
    if (!form.strategy_id || !form.start_date || !form.end_date) { alert("Strateji, başlangıç ve bitiş tarihi gerekli"); return; }
    setRunning(true);
    try {
      const { backtestsApi: btApi } = await import("../utils/api");
      const r = await btApi.create({ ...form, initial_balance: Number(form.initial_balance), start_date: new Date(form.start_date).toISOString(), end_date: new Date(form.end_date).toISOString() });
      const d = r.data;
      if (d.id) {
        loadBacktests();
        const poll = setInterval(async () => {
          const res = await btApi.get(d.id);
          const bt = res.data;
          if (bt.status === "completed" || bt.status === "failed") {
            clearInterval(poll);
            setRunning(false);
            loadBacktests();
            if (bt.status === "completed") setSelected(bt);
          }
        }, 3000);
      } else { setRunning(false); alert(d.detail || "Hata"); }
    } catch { setRunning(false); alert("Bağlantı hatası"); }
  };

  const inp = (label: string, key: string, type = "text", extra?: any) => (
    <div>
      <div style={{ fontSize: 10, color: C.muted, marginBottom: 3 }}>{label}</div>
      <input type={type} value={(form as any)[key]} onChange={e => setForm({ ...form, [key]: e.target.value })} {...extra}
        style={{ width: "100%", background: "#111827", border: `1px solid ${C.border}`, borderRadius: 4, padding: "6px 8px", color: C.text, fontSize: 12, outline: "none", boxSizing: "border-box" as any }} />
    </div>
  );

  const statusColor = (s: string) => s === "completed" ? C.green : s === "failed" ? C.red : s === "running" ? C.yellow : C.muted;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 16, height: "100%" }}>
      {/* Left: Form + List */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {/* New Backtest Form */}
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: C.text, marginBottom: 12 }}>🧪 Yeni Backtest</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {inp("İsim", "name")}
            <div>
              <div style={{ fontSize: 10, color: C.muted, marginBottom: 3 }}>Strateji</div>
              <select value={form.strategy_id} onChange={e => setForm({ ...form, strategy_id: e.target.value })}
                style={{ width: "100%", background: "#111827", border: `1px solid ${C.border}`, borderRadius: 4, padding: "6px 8px", color: C.text, fontSize: 12, outline: "none" }}>
                <option value="">Seç...</option>
                {strategies.map((s: any) => <option key={s.id} value={s.id}>{s.name}</option>)}
              </select>
            </div>
            <div>
              <div style={{ fontSize: 10, color: C.muted, marginBottom: 3 }}>Sembol</div>
              <select value={form.symbol} onChange={e => setForm({ ...form, symbol: e.target.value })}
                style={{ width: "100%", background: "#111827", border: `1px solid ${C.border}`, borderRadius: 4, padding: "6px 8px", color: C.text, fontSize: 12, outline: "none" }}>
                {symbols.length === 0 && <option value="">Yükleniyor...</option>}
                {symbols.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div>
              <div style={{ fontSize: 10, color: C.muted, marginBottom: 3 }}>Timeframe</div>
              <select value={form.timeframe} onChange={e => setForm({ ...form, timeframe: e.target.value })}
                style={{ width: "100%", background: "#111827", border: `1px solid ${C.border}`, borderRadius: 4, padding: "6px 8px", color: C.text, fontSize: 12, outline: "none" }}>
                {["1h","4h","1d"].map(tf => <option key={tf} value={tf}>{tf}</option>)}
              </select>
            </div>
            {inp("Başlangıç Tarihi", "start_date", "date")}
            {inp("Bitiş Tarihi", "end_date", "date")}
            {inp("Başlangıç Bakiyesi (EUR)", "initial_balance", "number")}
            <button onClick={run} disabled={running}
              style={{ background: running ? C.muted : C.accent, color: "white", border: "none", borderRadius: 6, padding: "9px", fontSize: 12, fontWeight: 700, cursor: running ? "not-allowed" : "pointer", marginTop: 4 }}>
              {running ? "⏳ Çalışıyor..." : "▶ Çalıştır"}
            </button>
          </div>
        </div>

        {/* Backtest List */}
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 14, flex: 1, overflowY: "auto" as any }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: C.text, marginBottom: 10 }}>Geçmiş Testler</div>
          {backtests.length === 0 && <div style={{ color: C.muted, fontSize: 11 }}>Henüz test yok</div>}
          {backtests.map((b: any) => (
            <div key={b.id} onClick={() => setSelected(b)} style={{ padding: "8px 10px", marginBottom: 6, borderRadius: 6, border: `1px solid ${selected?.id === b.id ? C.accent : C.border}`, cursor: "pointer", background: selected?.id === b.id ? "#111827" : "transparent" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: C.text }}>{b.name || b.symbol}</div>
                <div style={{ fontSize: 10, color: statusColor(b.status) }}>{b.status}</div>
              </div>
              <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>{b.symbol} · {b.timeframe}</div>
              {b.status === "completed" && (
                <div style={{ fontSize: 10, marginTop: 4, color: (b.total_return_pct ?? 0) >= 0 ? C.green : C.red }}>
                  {(b.total_return_pct ?? 0) >= 0 ? "+" : ""}{b.total_return_pct?.toFixed(2)}% · WR: {b.win_rate?.toFixed(1)}%
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Right: Results */}
      <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16, overflowY: "auto" as any }}>
        {!selected && <div style={{ color: C.muted, fontSize: 12, textAlign: "center", marginTop: 40 }}>Sonuçları görmek için bir test seçin</div>}
        {selected && selected.status === "failed" && <div style={{ color: C.red, fontSize: 12 }}>❌ Hata: {selected.error_message}</div>}
        {selected && selected.status === "completed" && (
          <>
            <div style={{ fontSize: 13, fontWeight: 700, color: C.text, marginBottom: 14 }}>{selected.name} — {selected.symbol} {selected.timeframe}</div>

            {/* Stats Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 16 }}>
              {[
                { label: "Net Getiri", value: `${(selected.total_return_pct ?? 0) >= 0 ? "+" : ""}${selected.total_return_pct?.toFixed(2)}%`, color: (selected.total_return_pct ?? 0) >= 0 ? C.green : C.red },
                { label: "Win Rate", value: `${selected.win_rate?.toFixed(1)}%`, color: C.text },
                { label: "Toplam Trade", value: selected.total_trades, color: C.text },
                { label: "Kazanan / Kaybeden", value: `${selected.winning_trades} / ${selected.losing_trades}`, color: C.text },
                { label: "Max Drawdown", value: `${selected.max_drawdown_pct?.toFixed(2)}%`, color: C.red },
                { label: "Sharpe Ratio", value: selected.sharpe_ratio?.toFixed(2), color: C.text },
                { label: "Profit Factor", value: selected.profit_factor?.toFixed(2), color: (selected.profit_factor ?? 0) >= 1 ? C.green : C.red },
                { label: "Final Bakiye", value: `${selected.final_balance?.toFixed(2)} EUR`, color: C.text },
              ].map((s: any) => (
                <div key={s.label} style={{ background: "#111827", borderRadius: 6, padding: "10px 12px", border: `1px solid ${C.border}` }}>
                  <div style={{ fontSize: 10, color: C.muted, marginBottom: 4 }}>{s.label}</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: s.color }}>{s.value}</div>
                </div>
              ))}
            </div>

            {/* Equity Curve */}
            {selected.equity_curve && selected.equity_curve.length > 0 && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: C.text, marginBottom: 8 }}>Equity Curve</div>
                <div style={{ height: 120, background: "#111827", borderRadius: 6, border: `1px solid ${C.border}`, position: "relative", overflow: "hidden" }}>
                  <svg width="100%" height="100%" viewBox={`0 0 ${selected.equity_curve.length} 100`} preserveAspectRatio="none">
                    {(() => {
                      const vals: number[] = selected.equity_curve;
                      const min = Math.min(...vals), max = Math.max(...vals), range = max - min || 1;
                      const pts = vals.map((v: number, i: number) => `${i},${100 - ((v - min) / range) * 90 - 5}`).join(" ");
                      const color = vals[vals.length - 1] >= vals[0] ? "#10b981" : "#f87171";
                      return <polyline points={pts} fill="none" stroke={color} strokeWidth="0.8" />;
                    })()}
                  </svg>
                </div>
              </div>
            )}

            {/* Trade Log */}
            {selected.trade_log && selected.trade_log.length > 0 && (
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: C.text, marginBottom: 8 }}>Trade Logu (ilk 50)</div>
                <div style={{ overflowX: "auto" as any }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
                    <thead>
                      <tr style={{ color: C.muted }}>
                        {["Yön", "Giriş", "Çıkış", "PnL", "Süre (bar)", "HTF Trend"].map(h => (
                          <th key={h} style={{ textAlign: "left", padding: "4px 8px", borderBottom: `1px solid ${C.border}` }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {selected.trade_log.slice(0, 50).map((t: any, i: number) => (
                        <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                          <td style={{ padding: "4px 8px", color: t.side === "buy" ? C.green : C.red }}>{t.side?.toUpperCase()}</td>
                          <td style={{ padding: "4px 8px", color: C.text }}>{t.entry_price?.toFixed(5)}</td>
                          <td style={{ padding: "4px 8px", color: C.text }}>{t.exit_price?.toFixed(5)}</td>
                          <td style={{ padding: "4px 8px", color: t.pnl >= 0 ? C.green : C.red }}>{t.pnl >= 0 ? "+" : ""}{t.pnl?.toFixed(2)}</td>
                          <td style={{ padding: "4px 8px", color: C.muted }}>{t.duration_bars}</td>
                          <td style={{ padding: "4px 8px", color: C.muted }}>{t.htf_trend}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────── PERFORMANCE PANEL ───────────────────────────
function PerformancePanel() {
  const C = { card: "#1a3248", border: "#1e3a5f", text: "#e2e8f0", muted: "#64748b", accent: "#2dd4bf", green: "#2dd4bf", red: "#f87171", yellow: "#fbbf24", surface: "#162840" };
  const [period, setPeriod] = useState("30d");
  const [stats, setStats] = useState<any>(null);
  const [curve, setCurve] = useState<any[]>([]);
  const [breakdown, setBreakdown] = useState<any>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const days = period === "7d" ? 7 : period === "30d" ? 30 : period === "90d" ? 90 : 365;
    Promise.all([
      tradesApi.getStats(period),
      import("../utils/api").then(m => m.default.get("/dashboard/equity-curve", { params: { days } })),
      import("../utils/api").then(m => m.default.get("/dashboard/market-breakdown", { params: { days } })),
    ]).then(([s, c, b]) => {
      setStats(s.data);
      setCurve(Array.isArray(c.data) ? c.data : []);
      setBreakdown(b.data || {});
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [period]);

  const statCard = (label: string, value: string | number, color = C.text) => (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: "14px 16px" }}>
      <div style={{ fontSize: 10, color: C.muted, letterSpacing: "0.06em", textTransform: "uppercase" as any, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color }}>{value}</div>
    </div>
  );

  const periods = ["7d", "30d", "90d", "1y"];

  if (loading) return <div style={{ padding: 40, textAlign: "center", color: C.muted }}>Loading...</div>;

  const pnls = curve.map((c: any) => c.pnl);
  const cumulative = curve.map((c: any) => c.cumulative);
  const maxCum = Math.max(...cumulative, 0);
  const minCum = Math.min(...cumulative, 0);
  const range = maxCum - minCum || 1;
  const H = 140;

  return (
    <div style={{ padding: 24 }}>
      {/* Period selector */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        {periods.map(p => (
          <button key={p} onClick={() => setPeriod(p)}
            style={{ padding: "6px 16px", borderRadius: 6, border: `1px solid ${period === p ? C.accent : C.border}`, background: period === p ? C.accent + "22" : "transparent", color: period === p ? C.accent : C.muted, fontWeight: period === p ? 700 : 400, fontSize: 12, cursor: "pointer" }}>
            {p}
          </button>
        ))}
      </div>

      {/* Stats grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 20 }}>
        {statCard("Total Trades", stats?.total_trades ?? "-")}
        {statCard("Win Rate", stats?.win_rate != null ? `${stats.win_rate.toFixed(1)}%` : "-", stats?.win_rate >= 50 ? C.green : C.red)}
        {statCard("Net P&L", stats?.total_pnl != null ? `${stats.total_pnl >= 0 ? "+" : ""}${stats.total_pnl.toFixed(2)} EUR` : "-", stats?.total_pnl >= 0 ? C.green : C.red)}
        {statCard("Profit Factor", stats?.profit_factor != null ? stats.profit_factor.toFixed(2) : "-", stats?.profit_factor >= 1 ? C.green : C.red)}
        {statCard("Sharpe Ratio", stats?.sharpe_ratio != null ? stats.sharpe_ratio.toFixed(2) : "-")}
        {statCard("Max Drawdown", stats?.max_drawdown != null ? `${stats.max_drawdown.toFixed(2)}%` : "-", C.red)}
        {statCard("Winners", stats?.winning_trades ?? "-", C.green)}
        {statCard("Losers", stats?.losing_trades ?? "-", C.red)}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
        {/* Equity Curve */}
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: C.text, marginBottom: 12 }}>Equity Curve</div>
          {curve.length === 0
            ? <div style={{ textAlign: "center", color: C.muted, fontSize: 12, padding: 40 }}>No trades in this period</div>
            : (
              <svg width="100%" height={H} viewBox={`0 0 ${curve.length} ${H}`} preserveAspectRatio="none">
                {(() => {
                  const pts = cumulative.map((v, i) => `${i},${H - ((v - minCum) / range) * (H - 10) - 5}`).join(" ");
                  const last = cumulative[cumulative.length - 1] ?? 0;
                  const color = last >= 0 ? "#2dd4bf" : "#f87171";
                  return <>
                    <polyline points={pts} fill="none" stroke={color} strokeWidth="1.2" />
                    <line x1="0" y1={H - ((0 - minCum) / range) * (H - 10) - 5} x2={curve.length} y2={H - ((0 - minCum) / range) * (H - 10) - 5} stroke="#1e3a5f" strokeWidth="0.5" strokeDasharray="4,4" />
                  </>;
                })()}
              </svg>
            )
          }
          {curve.length > 0 && (
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: C.muted, marginTop: 6 }}>
              <span>{curve[0]?.date}</span>
              <span>{curve[curve.length - 1]?.date}</span>
            </div>
          )}
        </div>

        {/* Market Breakdown */}
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: C.text, marginBottom: 12 }}>By Market</div>
          {Object.keys(breakdown).length === 0
            ? <div style={{ textAlign: "center", color: C.muted, fontSize: 12, padding: 20 }}>No data</div>
            : Object.entries(breakdown).map(([market, data]: [string, any]) => (
              <div key={market} style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: C.text, textTransform: "uppercase" as any }}>{market}</span>
                  <span style={{ fontSize: 11, color: data.pnl >= 0 ? C.green : C.red, fontWeight: 700 }}>
                    {data.pnl >= 0 ? "+" : ""}{data.pnl.toFixed(2)}
                  </span>
                </div>
                <div style={{ display: "flex", gap: 8, fontSize: 10, color: C.muted, marginBottom: 4 }}>
                  <span>{data.trades} trades</span>
                  <span>WR: {data.win_rate.toFixed(1)}%</span>
                </div>
                <div style={{ height: 4, background: "#1e3a5f", borderRadius: 2 }}>
                  <div style={{ width: `${data.win_rate}%`, height: "100%", background: data.win_rate >= 50 ? C.accent : C.red, borderRadius: 2 }} />
                </div>
              </div>
            ))
          }
        </div>
      </div>

      {/* Daily P&L Bar Chart */}
      {curve.length > 0 && (
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16, marginTop: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: C.text, marginBottom: 12 }}>Daily P&L</div>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 80 }}>
            {curve.slice(-60).map((d: any, i: number) => {
              const maxAbs = Math.max(...pnls.map(Math.abs), 1);
              const h = Math.abs(d.pnl) / maxAbs * 70;
              return (
                <div key={i} title={`${d.date}: ${d.pnl >= 0 ? "+" : ""}${d.pnl.toFixed(2)}`}
                  style={{ flex: 1, height: h, minHeight: 2, background: d.pnl >= 0 ? C.accent : C.red, borderRadius: 1, opacity: 0.85 }} />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}