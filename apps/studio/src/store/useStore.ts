import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { api } from '../lib/api';

export type EntityType = 'dataset' | 'job' | 'report' | 'label' | 'chat';

export interface Dataset {
  id: string;
  name: string;
  size: string;
  rows: number;
}

export interface Job {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
}

export interface Report {
  id: string;
  name: string;
  date: string;
  summary: string;
}

export interface LabelTask {
  id: string;
  name: string;
  type: 'text' | 'image_bbox';
  progress: number;
  total: number;
}

export interface Attachment {
  id: string;
  name: string;
  size: number;
  type: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  attachments?: Attachment[];
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: Date;
}

interface AppState {
  isAuthenticated: boolean;
  currentView: EntityType;
  selectedDatasetId: string | null;
  selectedJobId: string | null;
  selectedReportId: string | null;
  selectedLabelId: string | null;
  datasets: Dataset[];
  jobs: Job[];
  reports: Report[];
  labelTasks: LabelTask[];
  sessions: ChatSession[];
  availableModels: any[];
  activeSessionId: string;
  isTyping: boolean;
  isSidebarOpen: boolean;
  
  // Agent Chatbox Settings
  selectedModel: string;
  agentMode: 'plan' | 'build';
  
  login: (username: string, pass: string) => boolean;
  logout: () => void;
  setView: (view: EntityType, id?: string | null) => void;
  addMessage: (msg: Omit<Message, 'id' | 'timestamp'>) => void;
  updateLastMessage: (content: string) => void;
  setTyping: (status: boolean) => void;
  createNewSession: () => Promise<void>;
  deleteSession: (id: string) => Promise<void>;
  setActiveSession: (id: string) => void;
  toggleSidebar: () => void;
  
  setModel: (model: string) => void;
  setAgentMode: (mode: 'plan' | 'build') => void;
  initData: () => Promise<void>;
  loadSessionHistory: (sessionId: string) => Promise<void>;
}

export const useStore = create<AppState>()(
  persist(
    (set, get) => ({
      isAuthenticated: false,
      currentView: 'chat',
      selectedDatasetId: null,
      selectedJobId: null,
      selectedReportId: null,
      selectedLabelId: null,
      isTyping: false,
      isSidebarOpen: window.innerWidth >= 1024,
      
      selectedModel: 'GPT-4',
      agentMode: 'build',
      availableModels: [],

      datasets: [],
      jobs: [],
      reports: [],
      labelTasks: [],
      sessions: [
        {
          id: 'session-1',
          title: 'New Chat',
          messages: [
            {
              id: '1',
              role: 'assistant',
              content: 'Welcome to Ads Studio. How can I assist with your ML workflows today? Try typing `/` for commands.',
              timestamp: new Date()
            }
          ],
          updatedAt: new Date()
        }
      ],
      activeSessionId: 'session-1',
      
      login: (username, pass) => {
        if (username === 'guest' && pass === '12345') {
          set({ isAuthenticated: true });
          get().initData();
          return true;
        }
        return false;
      },
      logout: () => set({ isAuthenticated: false }),
      
      initData: async () => {
        try {
          // Fetch baseline data from API
          const [datasets, jobs, labels, reports, models, sessionsList] = await Promise.all([
            api.datasets.list().catch(() => []),
            api.jobs.list().catch(() => []),
            api.labels.list().catch(() => []),
            api.reports.list().catch(() => []),
            api.chat.getModels().catch(() => []),
            api.chat.getSessions().catch(() => [])
          ]);
          
          set((state) => {
            const updates: any = {
              datasets: datasets.length ? datasets.map((d: any) => ({ ...d, size: 'Unknown', rows: 0 })) : state.datasets,
              jobs: jobs.length ? jobs.map((j: any) => ({ ...j, status: j.status || 'completed', progress: j.progress || 100 })) : state.jobs,
              labelTasks: labels.length ? labels.map((l: any) => ({ ...l, type: l.type || 'text', progress: 0, total: 100 })) : state.labelTasks,
              reports: reports.length ? reports.map((r: any) => ({ ...r, date: 'Recent', summary: r.description })) : state.reports,
              availableModels: models.length ? models : state.availableModels,
              selectedModel: models.length ? models[0].id : state.selectedModel,
            };

            // Map backend sessions to our local state
            if (sessionsList && sessionsList.length > 0) {
              updates.sessions = sessionsList.map((s: any) => {
                const existing = state.sessions.find(es => es.id === s.id);
                return {
                  id: s.id,
                  title: s.name || s.id,
                  messages: existing ? existing.messages : [], // Keep local messages temporarily until history loads
                  updatedAt: new Date(s.updated_at || Date.now())
                };
              });
              
              // Set the active session to the first one if the current active isn't in the list
              if (!updates.sessions.find((s: any) => s.id === state.activeSessionId)) {
                updates.activeSessionId = updates.sessions[0].id;
              }
            }

            return updates;
          });

          // Fetch history for the active session right after initializing
          const currentActiveSession = get().activeSessionId;
          if (currentActiveSession) {
            await get().loadSessionHistory(currentActiveSession);
          }

        } catch (error) {
          console.error('Failed to initialize data from API', error);
        }
      },

      loadSessionHistory: async (sessionId: string) => {
        try {
          const historyData = await api.chat.getHistory(sessionId);
          
          if (historyData && historyData.messages) {
            set((state) => {
              const updatedSessions = state.sessions.map(s => {
                if (s.id === sessionId) {
                  return {
                    ...s,
                    messages: historyData.messages.map((m: any, idx: number) => ({
                      id: m.id || `${Date.now()}-${idx}`,
                      role: m.role || 'user',
                      content: m.content || '',
                      timestamp: new Date()
                    }))
                  };
                }
                return s;
              });

              return {
                sessions: updatedSessions,
                selectedModel: historyData.model || state.selectedModel,
                agentMode: historyData.mode || state.agentMode,
              };
            });
          }
        } catch (error) {
          console.error(`Failed to load history for session ${sessionId}`, error);
        }
      },

      setView: (view, id) => {
        set(() => {
          const updates: any = { currentView: view };
          if (id !== undefined) {
            if (view === 'dataset') updates.selectedDatasetId = id;
            if (view === 'job') updates.selectedJobId = id;
            if (view === 'report') updates.selectedReportId = id;
            if (view === 'label') updates.selectedLabelId = id;
          }
          if (window.innerWidth < 1024) {
            updates.isSidebarOpen = false;
          }
          return updates;
        });
      },
      
      createNewSession: async () => {
        try {
          const res = await api.chat.createNewSession();
          const newId = res.new_session_id;
          set((state) => {
            const newSession: ChatSession = {
              id: newId,
              title: 'New Chat',
              messages: [{
                id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
                role: 'assistant',
                content: 'Welcome to Ads Studio. How can I assist with your ML workflows today?',
                timestamp: new Date()
              }],
              updatedAt: new Date()
            };
            return {
              sessions: [newSession, ...state.sessions],
              activeSessionId: newId,
              currentView: 'chat',
              isSidebarOpen: window.innerWidth >= 1024 ? state.isSidebarOpen : false
            };
          });
        } catch (error) {
          console.error("Failed to create session", error);
        }
      },

      deleteSession: async (id: string) => {
        try {
          await api.chat.deleteSession(id);
          
          set((state) => {
            const filteredSessions = state.sessions.filter(s => s.id !== id);
            const updates: any = { sessions: filteredSessions };
            
            // If we deleted the active session, switch to another one
            if (state.activeSessionId === id) {
              if (filteredSessions.length > 0) {
                updates.activeSessionId = filteredSessions[0].id;
              } else {
                updates.activeSessionId = '';
              }
            }
            return updates;
          });

          // Fetch the history for the new active session, or create a new session if empty
          const { activeSessionId, createNewSession, loadSessionHistory } = get();
          if (!activeSessionId) {
            await createNewSession();
          } else {
            await loadSessionHistory(activeSessionId);
          }
        } catch (error) {
          console.error("Failed to delete session", error);
        }
      },

      setActiveSession: (id) => {
        set((state) => ({ 
          activeSessionId: id, 
          currentView: 'chat',
          isSidebarOpen: window.innerWidth >= 1024 ? state.isSidebarOpen : false
        }));
        // Load history automatically when switching to a session
        get().loadSessionHistory(id);
      },
      
      addMessage: (msg) => set((state) => {
        const uniqueId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        const newMsg = { ...msg, id: uniqueId, timestamp: new Date() };
        
        const updatedSessions = state.sessions.map(s => {
          if (s.id === state.activeSessionId) {
            const title = s.messages.length === 1 && msg.role === 'user' && msg.content
              ? (msg.content.length > 20 ? msg.content.slice(0, 20) + '...' : msg.content)
              : s.title;
            return { ...s, title, messages: [...s.messages, newMsg], updatedAt: new Date() };
          }
          return s;
        });
        return { sessions: updatedSessions };
      }),

      updateLastMessage: (content) => set((state) => {
        const updatedSessions = state.sessions.map(s => {
          if (s.id === state.activeSessionId) {
            const msgs = [...s.messages];
            if (msgs.length > 0 && msgs[msgs.length - 1].role === 'assistant') {
              msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content };
            }
            return { ...s, messages: msgs };
          }
          return s;
        });
        return { sessions: updatedSessions };
      }),
      
      setTyping: (status) => set({ isTyping: status }),
      
      toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
      
      setModel: (model) => set({ selectedModel: model }),
      setAgentMode: (mode) => set({ agentMode: mode }),
    }),
    {
      name: 'ads-studio-storage',
      partialize: (state) => ({
        isAuthenticated: state.isAuthenticated,
        // Since we are loading true data from API, we don't strictly need to persist all session data, 
        // but persisting helps UI feel fast before API finishes.
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
        currentView: state.currentView,
        selectedDatasetId: state.selectedDatasetId,
        selectedJobId: state.selectedJobId,
        selectedReportId: state.selectedReportId,
        selectedLabelId: state.selectedLabelId,
        selectedModel: state.selectedModel,
        agentMode: state.agentMode,
      }),
    }
  )
);