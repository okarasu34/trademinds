import axios, { AxiosInstance } from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// ─── Request interceptor: attach access token ───
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// ─── Response interceptor: handle 401 + refresh ───
api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const original = error.config;
    if (error.response?.status === 401 && !original._retry) {
      original._retry = true;
      try {
        const refresh = localStorage.getItem("refresh_token");
        const res = await axios.post(`${BASE_URL}/auth/refresh`, { refresh_token: refresh });
        const { access_token, refresh_token } = res.data;
        localStorage.setItem("access_token", access_token);
        localStorage.setItem("refresh_token", refresh_token);
        original.headers.Authorization = `Bearer ${access_token}`;
        return api(original);
      } catch {
        localStorage.clear();
        window.location.href = "/login";
      }
    }
    return Promise.reject(error);
  }
);

// ─── API Methods ───

export const authApi = {
  login: (email: string, password: string, totp_code?: string) =>
    api.post("/auth/login", { email, password, totp_code }),
  register: (email: string, password: string) =>
    api.post("/auth/register", { email, password }),
  logout: (refresh_token: string) =>
    api.post("/auth/logout", { refresh_token }),
  me: () => api.get("/auth/me"),
  setup2fa: () => api.get("/auth/2fa/setup"),
  enable2fa: (totp_code: string) => api.post("/auth/2fa/enable", { totp_code }),
};

export const botApi = {
  getStatus: () => api.get("/bot/status"),
  start: () => api.post("/bot/start"),
  stop: () => api.post("/bot/stop"),
  pause: () => api.post("/bot/pause"),
  updateConfig: (data: object) => api.put("/bot/config", data),
  setMode: (mode: string) => api.put("/bot/mode", { mode }),
  getHealthLogs: () => api.get("/bot/health-logs"),
  resetDaily: () => api.post("/bot/reset-daily"),
};

export const tradesApi = {
  list: (params?: object) => api.get("/trades", { params }),
  getOpen: () => api.get("/trades/open"),
  getStats: (period: string) => api.get("/trades/stats", { params: { period } }),
  get: (id: string) => api.get(`/trades/${id}`),
  manualClose: (id: string) => api.post(`/trades/${id}/close`),
};

export const strategiesApi = {
  list: () => api.get("/strategies"),
  create: (data: object) => api.post("/strategies", data),
  get: (id: string) => api.get(`/strategies/${id}`),
  update: (id: string, data: object) => api.put(`/strategies/${id}`, data),
  delete: (id: string) => api.delete(`/strategies/${id}`),
  toggle: (id: string) => api.post(`/strategies/${id}/toggle`),
};

export const brokersApi = {
  list: () => api.get("/brokers"),
  add: (data: object) => api.post("/brokers", data),
  remove: (id: string) => api.delete(`/brokers/${id}`),
  test: (id: string) => api.post(`/brokers/${id}/test`),
};

export const reportsApi = {
  downloadPdf: (period: string) =>
    api.get("/reports/pdf", { params: { period }, responseType: "blob" }),
  downloadExcel: (period: string) =>
    api.get("/reports/excel", { params: { period }, responseType: "blob" }),
};

export const calendarApi = {
  get: (params?: object) => api.get("/calendar", { params }),
  getHighImpact: (minutes?: number) =>
    api.get("/calendar/upcoming-high-impact", { params: { minutes } }),
};

export const dashboardApi = {
  getSummary: () => api.get("/dashboard/summary"),
};

export default api;
