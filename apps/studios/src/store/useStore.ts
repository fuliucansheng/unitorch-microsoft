import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { api } from '../lib/api';

export type EntityType = 'dataset' | 'job' | 'report' | 'label' | 'chat' | 'settings';

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

export interface UserProfile {
  username: string;
  email: string;
  fullName: string;
  avatar: string;
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
  entities: any[];
  activeSessionId: string;
  isTyping: boolean;
  isSidebarOpen: boolean;
  
  userProfile: UserProfile;
  
  // Agent Chatbox Settings
  selectedModel: string;
  agentMode: 'plan' | 'build';
  
  login: (username: string, pass: string) => boolean;
  logout: () => void;
  setView: (view: EntityType, id?: string | null) => void;
  updateProfile: (profile: Partial<UserProfile>) => void;
  addMessage: (msg: Omit<Message, 'id' | 'timestamp'>) => void;
  updateLastMessage: (content: string) => void;
  setTyping: (status: boolean) => void;
  createNewSession: () => Promise<void>;
  deleteSession: (id: string) => Promise<void>;
  renameSession: (id: string, name: string) => Promise<void>;
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
      
      userProfile: {
        username: 'guest',
        email: 'guest@example.com',
        fullName: 'Guest User',
        avatar: '',
      },
      
      selectedModel: 'gemini-3.1-pro-preview',
      agentMode: 'build',
      availableModels: [],
      entities: [],

      datasets: [],
      jobs: [],
      reports: [],
      labelTasks: [],
      sessions: [
        {
          id: 'session-1',
          title: 'New Chat',
          messages: [],
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
      
      updateProfile: (profile) => set((state) => ({
        userProfile: { ...state.userProfile, ...profile }
      })),
      
      initData: async () => {
        try {
          // Fetch baseline data from API. We use null to differentiate between API error and empty array [].
          const [datasets, jobs, labels, reports, models, sessionsList, entities] = await Promise.all([
            api.datasets.list().catch(() => null),
            api.jobs.list().catch(() => null),
            api.labels.list().catch(() => null),
            api.reports.list().catch(() => null),
            api.chat.getModels().catch(() => null),
            api.chat.getSessions().catch(() => null),
            api.chat.getEntities().catch(() => null)
          ]);
          
          set((state) => {
            const updates: any = {};

            // If API returned an array (even empty), we update the state to match it exactly.
            if (datasets !== null) updates.datasets = datasets.map((d: any) => ({ ...d, size: 'Unknown', rows: 0 }));
            if (jobs !== null) updates.jobs = jobs.map((j: any) => ({ ...j, status: j.status || 'completed', progress: j.progress || 100 }));
            if (labels !== null) updates.labelTasks = labels.map((l: any) => ({ ...l, type: l.type || 'text', progress: 0, total: 100 }));
            if (reports !== null) updates.reports = reports.map((r: any) => ({ ...r, date: 'Recent', summary: r.description }));
            if (models !== null) updates.availableModels = models;
            if (entities !== null) updates.entities = entities;

            if (models !== null && models.length > 0) {
              const modelExists = models.find((m: any) => m.id === state.selectedModel);
              if (!modelExists) {
                const defaultModel = models.find((m: any) => m.id === 'gemini-3.1-pro-preview');
                updates.selectedModel = defaultModel ? defaultModel.id : models[0].id;
              }
            }

            // Map backend sessions to our local state
            if (sessionsList !== null) {
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
              if (updates.sessions.length > 0 && !updates.sessions.find((s: any) => s.id === state.activeSessionId)) {
                updates.activeSessionId = updates.sessions[0].id;
              } else if (updates.sessions.length === 0) {
                updates.activeSessionId = '';
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
                agentMode: (historyData.mode === 'plan' || historyData.mode === 'build') ? historyData.mode : 'build', 
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
              messages: [],
              updatedAt: new Date()
            };
            return {
              sessions: [newSession, ...state.sessions],
              activeSessionId: newId,
              currentView: 'chat',
              agentMode: 'build', // Force to build for new chat
              isSidebarOpen: window.innerWidth >= 1024 ? state.isSidebarOpen : false
            };
          });
          
          // Fetch history immediately after creating
          await get().loadSessionHistory(newId);
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

      renameSession: async (id: string, name: string) => {
        try {
          await api.chat.renameSession(id, name);
          set((state) => ({
            sessions: state.sessions.map(s => s.id === id ? { ...s, title: name } : s)
          }));
        } catch (error) {
          console.error("Failed to rename session", error);
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
        userProfile: state.userProfile,
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
        currentView: state.currentView,
        selectedDatasetId: state.selectedDatasetId,
        selectedJobId: state.selectedJobId,
        selectedReportId: state.selectedReportId,
        selectedLabelId: state.selectedLabelId,
        selectedModel: state.selectedModel,
      }),
    }
  )
);