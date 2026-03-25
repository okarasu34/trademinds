import { create } from "zustand";
import { persist } from "zustand/middleware";

// ─── Auth Store ───

interface AuthState {
  user: { id: string; email: string; is_2fa_enabled: boolean; base_currency: string } | null;
  accessToken: string | null;
  refreshToken: string | null;
  setAuth: (user: AuthState["user"], access: string, refresh: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      setAuth: (user, accessToken, refreshToken) =>
        set({ user, accessToken, refreshToken }),
      logout: () => set({ user: null, accessToken: null, refreshToken: null }),
    }),
    { name: "trademinds-auth" }
  )
);

// ─── Bot Store ───

interface BotState {
  status: "running" | "paused" | "stopped" | "error";
  tradeMode: "paper" | "live" | "backtest";
  openPositions: number;
  dailyPnl: number;
  dailyTrades: number;
  maxPositions: number;
  currency: string;
  setBotStatus: (status: BotState["status"]) => void;
  setTradeMode: (mode: BotState["tradeMode"]) => void;
  setOpenPositions: (n: number) => void;
  setDailyPnl: (pnl: number) => void;
  setCurrency: (currency: string) => void;
  updateFromApi: (data: Partial<BotState>) => void;
}

export const useBotStore = create<BotState>()((set) => ({
  status: "stopped",
  tradeMode: "paper",
  openPositions: 0,
  dailyPnl: 0,
  dailyTrades: 0,
  maxPositions: 25,
  currency: "USD",
  setBotStatus: (status) => set({ status }),
  setTradeMode: (tradeMode) => set({ tradeMode }),
  setOpenPositions: (openPositions) => set({ openPositions }),
  setDailyPnl: (dailyPnl) => set({ dailyPnl }),
  setCurrency: (currency) => set({ currency }),
  updateFromApi: (data) => set((state) => ({ ...state, ...data })),
}));

// ─── Prices Store ───

interface Price {
  price: number;
  bid: number;
  ask: number;
  change: number;
}

interface PricesState {
  prices: Record<string, Price>;
  updatePrice: (symbol: string, data: Price) => void;
  setAllPrices: (prices: Record<string, Price>) => void;
}

export const usePricesStore = create<PricesState>()((set) => ({
  prices: {},
  updatePrice: (symbol, data) =>
    set((state) => ({ prices: { ...state.prices, [symbol]: data } })),
  setAllPrices: (prices) => set({ prices }),
}));

// ─── Trade Store ───

export interface Trade {
  id: string;
  symbol: string;
  market_type: string;
  side: string;
  status: string;
  entry_price: number;
  exit_price?: number;
  lot_size: number;
  stop_loss?: number;
  take_profit?: number;
  pnl?: number;
  pnl_pct?: number;
  ai_reasoning?: string;
  ai_confidence?: number;
  strategy_name?: string;
  opened_at: string;
  closed_at?: string;
  closed_by?: string;
}

interface TradesState {
  openTrades: Trade[];
  recentTrades: Trade[];
  setOpenTrades: (trades: Trade[]) => void;
  setRecentTrades: (trades: Trade[]) => void;
  addTrade: (trade: Trade) => void;
  closeTrade: (tradeId: string, pnl: number) => void;
}

export const useTradesStore = create<TradesState>()((set) => ({
  openTrades: [],
  recentTrades: [],
  setOpenTrades: (openTrades) => set({ openTrades }),
  setRecentTrades: (recentTrades) => set({ recentTrades }),
  addTrade: (trade) =>
    set((state) => ({
      openTrades: [trade, ...state.openTrades],
    })),
  closeTrade: (tradeId, pnl) =>
    set((state) => ({
      openTrades: state.openTrades.filter((t) => t.id !== tradeId),
      recentTrades: [
        { ...state.openTrades.find((t) => t.id === tradeId)!, pnl, status: "closed" },
        ...state.recentTrades.slice(0, 49),
      ],
    })),
}));

// ─── UI Store ───

interface UIState {
  sidebarOpen: boolean;
  activeTab: string;
  notification: { message: string; type: "success" | "error" | "info" } | null;
  setActiveTab: (tab: string) => void;
  showNotification: (message: string, type?: UIState["notification"]["type"]) => void;
  clearNotification: () => void;
}

export const useUIStore = create<UIState>()((set) => ({
  sidebarOpen: true,
  activeTab: "dashboard",
  notification: null,
  setActiveTab: (activeTab) => set({ activeTab }),
  showNotification: (message, type = "success") => {
    set({ notification: { message, type } });
    setTimeout(() => set({ notification: null }), 4000);
  },
  clearNotification: () => set({ notification: null }),
}));
