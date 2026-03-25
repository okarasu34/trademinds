import { useState } from "react";
import { useAuthStore } from "../store";
import { authApi } from "../utils/api";

export default function Login() {
  const { setAuth } = useAuthStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [totpCode, setTotpCode] = useState("");
  const [needs2fa, setNeeds2fa] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    setError("");
    setLoading(true);
    try {
      const res = await authApi.login(email, password, needs2fa ? totpCode : undefined);
      const { access_token, refresh_token, user } = res.data;
      localStorage.setItem("access_token", access_token);
      localStorage.setItem("refresh_token", refresh_token);
      setAuth(user, access_token, refresh_token);
      window.location.href = "/";
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      if (detail === "2FA_REQUIRED") {
        setNeeds2fa(true);
        setError("");
      } else {
        setError(detail || "Login failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const S = {
    page: { minHeight: "100vh", background: "#080c14", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'IBM Plex Mono', monospace" } as const,
    card: { background: "#0d1421", border: "1px solid #1e2d45", borderRadius: 12, padding: 40, width: 380 } as const,
    title: { fontSize: 22, fontWeight: 800, color: "#f1f5f9", marginBottom: 4 } as const,
    sub: { fontSize: 12, color: "#64748b", marginBottom: 32 } as const,
    label: { fontSize: 11, fontWeight: 600, color: "#94a3b8", marginBottom: 6, display: "block", textTransform: "uppercase", letterSpacing: "0.06em" } as const,
    input: { width: "100%", background: "#111827", border: "1px solid #1e2d45", borderRadius: 8, padding: "12px 14px", color: "#f1f5f9", fontSize: 13, fontFamily: "inherit", outline: "none", boxSizing: "border-box" } as const,
    btn: { width: "100%", background: "#3b82f6", color: "white", border: "none", borderRadius: 8, padding: "13px 0", fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: "inherit", marginTop: 24 } as const,
    error: { background: "#7f1d1d", color: "#fca5a5", padding: "10px 14px", borderRadius: 8, fontSize: 12, marginBottom: 16 } as const,
  };

  return (
    <div style={S.page}>
      <div style={S.card}>
        <div style={S.title}><span style={{ color: "#3b82f6" }}>Trade</span>Minds</div>
        <div style={S.sub}>AI Autonomous Trading System</div>

        {error && <div style={S.error}>{error}</div>}

        {!needs2fa ? (
          <>
            <div style={{ marginBottom: 16 }}>
              <label style={S.label}>Email</label>
              <input style={S.input} type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="your@email.com" onKeyDown={e => e.key === "Enter" && handleLogin()} />
            </div>
            <div>
              <label style={S.label}>Password</label>
              <input style={S.input} type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="••••••••" onKeyDown={e => e.key === "Enter" && handleLogin()} />
            </div>
          </>
        ) : (
          <div>
            <label style={S.label}>2FA Code</label>
            <input style={{ ...S.input, letterSpacing: "0.3em", fontSize: 20, textAlign: "center" }} type="text" value={totpCode} onChange={e => setTotpCode(e.target.value.replace(/\D/g, "").slice(0, 6))} placeholder="000000" maxLength={6} onKeyDown={e => e.key === "Enter" && handleLogin()} autoFocus />
            <div style={{ fontSize: 11, color: "#64748b", marginTop: 8 }}>Enter the 6-digit code from your authenticator app</div>
          </div>
        )}

        <button style={{ ...S.btn, opacity: loading ? 0.7 : 1 }} onClick={handleLogin} disabled={loading}>
          {loading ? "Signing in..." : needs2fa ? "Verify" : "Sign In"}
        </button>

        {needs2fa && (
          <button onClick={() => { setNeeds2fa(false); setTotpCode(""); }} style={{ width: "100%", background: "transparent", color: "#64748b", border: "none", marginTop: 12, cursor: "pointer", fontSize: 12, fontFamily: "inherit" }}>
            ← Back
          </button>
        )}
      </div>
    </div>
  );
}
