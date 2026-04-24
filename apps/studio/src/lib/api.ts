const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

// 基础的 fetch 封装
async function fetchApi<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  if (!response.ok) {
    throw new Error(`API Request failed: ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  // Utils
  utils: {
    upload: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_BASE_URL}/microsoft/apps/studio/utils/upload`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          // Note: Do not set Content-Type here, let the browser set it with the correct boundary for multipart/form-data
        },
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
      return response.json() as Promise<{ path: string; filename: string; size: number }>;
    }
  },

  // Chat
  chat: {
    getCommands: () => fetchApi<any[]>('/microsoft/apps/studio/chat/commands'),
    getEntities: () => fetchApi<any[]>('/microsoft/apps/studio/chat/entities'),
    getModels: () => fetchApi<any[]>('/microsoft/apps/studio/chat/models'),
    getSessions: () => fetchApi<any[]>('/microsoft/apps/studio/chat/sessions'),
    getHistory: (sessionId: string) => fetchApi<any>(`/microsoft/apps/studio/chat/history?session_id=${sessionId}`),
    createNewSession: (sessionId?: string) => 
      fetchApi<{new_session_id: string}>('/microsoft/apps/studio/chat/new', {
        method: 'POST',
        body: JSON.stringify(sessionId ? { session_id: sessionId } : {})
      }),
    deleteSession: (sessionId: string) => 
      fetchApi<any>('/microsoft/apps/studio/chat/delete', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId })
      }),
    // 流式响应会在组件/Store中单独处理
    getCompletionsUrl: () => `${API_BASE_URL}/microsoft/apps/studio/chat/completions`
  },

  // Datasets
  datasets: {
    list: () => fetchApi<any[]>('/microsoft/apps/studio/datasets'),
    getDetails: (id: string) => fetchApi<any>('/microsoft/apps/studio/datasets/details', {
      method: 'POST',
      body: JSON.stringify({ id })
    }),
    getPreview: (id: string, split: string = 'train', start: number = 0, limit: number = 5) => 
      fetchApi<any>('/microsoft/apps/studio/datasets/preview', {
        method: 'POST',
        body: JSON.stringify({ id, split, start, limit })
      })
  },

  // Jobs
  jobs: {
    list: () => fetchApi<any[]>('/microsoft/apps/studio/jobs'),
    getDetails: (id: string) => fetchApi<any>('/microsoft/apps/studio/jobs/details', {
      method: 'POST',
      body: JSON.stringify({ id })
    }),
    cancel: (id: string) => fetchApi<any>('/microsoft/apps/studio/jobs/cancel', {
      method: 'POST',
      body: JSON.stringify({ id })
    }),
    restart: (id: string) => fetchApi<any>('/microsoft/apps/studio/jobs/restart', {
      method: 'POST',
      body: JSON.stringify({ id })
    })
  },

  // Labels
  labels: {
    list: () => fetchApi<any[]>('/microsoft/apps/studio/labels'),
    getDetails: (id: string) => fetchApi<any>('/microsoft/apps/studio/labels/details', {
      method: 'POST',
      body: JSON.stringify({ id })
    })
  },

  // Reports
  reports: {
    list: () => fetchApi<any[]>('/microsoft/apps/studio/reports'),
    getDetails: (id: string) => fetchApi<any>('/microsoft/apps/studio/reports/details', {
      method: 'POST',
      body: JSON.stringify({ id })
    })
  }
};