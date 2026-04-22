import { create } from 'zustand';
import { persist } from 'zustand/middleware';

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
  status: 'running' | 'completed' | 'failed';
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
  selectedEntityId: string | null;
  datasets: Dataset[];
  jobs: Job[];
  reports: Report[];
  labelTasks: LabelTask[];
  sessions: ChatSession[];
  activeSessionId: string;
  isTyping: boolean;
  isSidebarOpen: boolean;
  
  login: (username: string, pass: string) => boolean;
  logout: () => void;
  setView: (view: EntityType, id?: string | null) => void;
  addMessage: (msg: Omit<Message, 'id' | 'timestamp'>) => void;
  setTyping: (status: boolean) => void;
  createNewSession: () => void;
  setActiveSession: (id: string) => void;
  toggleSidebar: () => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      isAuthenticated: false,
      currentView: 'chat',
      selectedEntityId: null,
      isTyping: false,
      isSidebarOpen: window.innerWidth >= 1024,
      datasets: [
        { id: 'data1', name: 'customer_reviews.csv', size: '24 MB', rows: 15420 },
        { id: 'data2', name: 'training_split_v2.json', size: '1.2 GB', rows: 250000 },
      ],
      jobs: [
        { id: 'job1', name: 'Fine-tune Llama-3', status: 'running', progress: 45 },
        { id: 'job2', name: 'Data preprocessing', status: 'completed', progress: 100 },
      ],
      reports: [
        { id: 'rep1', name: 'Model Evaluation v2', date: '2023-10-25', summary: 'Shows 15% improvement in F1 score.' }
      ],
      labelTasks: [
        { id: 'lbl1', name: 'Sentiment Analysis', type: 'text', progress: 342, total: 1000 },
        { id: 'lbl2', name: 'Vehicle Detection', type: 'image_bbox', progress: 89, total: 500 }
      ],
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
          return true;
        }
        return false;
      },
      logout: () => set({ isAuthenticated: false }),
      
      setView: (view, id = null) => {
        set({ currentView: view, selectedEntityId: id });
        if (window.innerWidth < 1024) {
          set({ isSidebarOpen: false });
        }
      },
      
      createNewSession: () => set((state) => {
        const newSession: ChatSession = {
          id: Date.now().toString(),
          title: 'New Chat',
          messages: [
            {
              id: Date.now().toString(),
              role: 'assistant',
              content: 'Welcome to Ads Studio. How can I assist with your ML workflows today?',
              timestamp: new Date()
            }
          ],
          updatedAt: new Date()
        };
        return {
          sessions: [newSession, ...state.sessions],
          activeSessionId: newSession.id,
          currentView: 'chat',
          selectedEntityId: null,
          isSidebarOpen: window.innerWidth >= 1024 ? state.isSidebarOpen : false
        };
      }),

      setActiveSession: (id) => set((state) => ({ 
        activeSessionId: id, 
        currentView: 'chat',
        selectedEntityId: null,
        isSidebarOpen: window.innerWidth >= 1024 ? state.isSidebarOpen : false
      })),
      
      addMessage: (msg) => set((state) => {
        const newMsg = { ...msg, id: Date.now().toString(), timestamp: new Date() };
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
      
      setTyping: (status) => set({ isTyping: status }),
      
      toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
    }),
    {
      name: 'ads-studio-storage',
      // 只持久化登录状态、当前视图和聊天记录，避免因为窗口大小引起的 UI 异常
      partialize: (state) => ({
        isAuthenticated: state.isAuthenticated,
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
        currentView: state.currentView,
        selectedEntityId: state.selectedEntityId,
      }),
    }
  )
);