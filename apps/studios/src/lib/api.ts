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
      
      const response = await fetch(`${API_BASE_URL}/microsoft/apps/studios/utils/upload`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
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
    getCommands: () => fetchApi<any[]>('/microsoft/apps/studios/chats/commands'),
    getEntities: () => fetchApi<any[]>('/microsoft/apps/studios/chats/entities'),
    getModels: () => fetchApi<any[]>('/microsoft/apps/studios/chats/models'),
    getSessions: () => fetchApi<any[]>('/microsoft/apps/studios/chats/sessions'),
    getHistory: (sessionId: string) => fetchApi<any>(`/microsoft/apps/studios/chats/history?session_id=${sessionId}`),
    createNewSession: (sessionId?: string) => 
      fetchApi<{new_session_id: string}>('/microsoft/apps/studios/chats/new', {
        method: 'POST',
        body: JSON.stringify(sessionId ? { session_id: sessionId } : {})
      }),
    deleteSession: (sessionId: string) => 
      fetchApi<any>('/microsoft/apps/studios/chats/delete', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId })
      }),
    renameSession: (sessionId: string, name: string) => 
      fetchApi<{session_id: string; name: string}>('/microsoft/apps/studios/chats/name', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, name })
      }),
    getCompletionsUrl: () => `${API_BASE_URL}/microsoft/apps/studios/chats/completions`
  },

  // Datasets
  datasets: {
    list: () => fetchApi<any[]>('/microsoft/apps/studios/datasets'),
    getDetails: (id: string) => fetchApi<any>('/microsoft/apps/studios/datasets/get', {
      method: 'POST',
      body: JSON.stringify({ id })
    }),
    updateMeta: (id: string, meta: any) => fetchApi<any>('/microsoft/apps/studios/datasets/meta/update', {
      method: 'POST',
      body: JSON.stringify({ id, meta })
    }),
    getPreview: (dataset_id: string, split?: string, start: number = 0, limit: number = 20) => {
      const body: any = { dataset_id, start, limit, include_meta: true };
      if (split) body.split = split;
      return fetchApi<any>('/microsoft/apps/studios/datasets/sample/list', {
        method: 'POST',
        body: JSON.stringify(body)
      });
    },
    // Assuming /datasets/delete exists based on the pattern, and keeping sample/meta/delete for specific sample updates
    delete: (id: string) => fetchApi<any>('/microsoft/apps/studios/datasets/delete', {
      method: 'POST',
      body: JSON.stringify({ id })
    }),
    deleteSampleMeta: (dataset_id: string, sample_ids: string[], keys?: string[]) => fetchApi<any>('/microsoft/apps/studios/datasets/sample/meta/delete', {
      method: 'POST',
      body: JSON.stringify({ dataset_id, sample_ids, keys })
    })
  },

  // Jobs
  jobs: {
    list: () => fetchApi<any[]>('/microsoft/apps/studios/jobs'),
    getDetails: (id: string) => fetchApi<any>('/microsoft/apps/studios/jobs/get', {
      method: 'POST',
      body: JSON.stringify({ id })
    }),
    getLogs: (id: string, tail: number | null = 100) => fetchApi<any>('/microsoft/apps/studios/jobs/logs', {
      method: 'POST',
      body: JSON.stringify({ id, tail })
    }),
    cancel: (id: string) => fetchApi<any>('/microsoft/apps/studios/jobs/cancel', {
      method: 'POST',
      body: JSON.stringify({ id })
    }),
    restart: (id: string) => fetchApi<any>('/microsoft/apps/studios/jobs/restart', {
      method: 'POST',
      body: JSON.stringify({ id })
    }),
    delete: (id: string) => fetchApi<any>('/microsoft/apps/studios/jobs/delete', {
      method: 'POST',
      body: JSON.stringify({ id })
    })
  },

  // Labels
  labels: {
    list: () => fetchApi<any[]>('/microsoft/apps/studios/labels'),
    getDetails: (id: string) => fetchApi<any>('/microsoft/apps/studios/labels/get', {
      method: 'POST',
      body: JSON.stringify({ id })
    }),
    update: (data: { id: string; name?: string; description?: string; ui_html?: string; display_fields?: any[]; label_fields?: any[]; extra?: any }) => fetchApi<any>('/microsoft/apps/studios/labels/update', {
      method: 'POST',
      body: JSON.stringify(data)
    }),
    getRandomSample: (task_id: string, sample_id: string | null = null) => fetchApi<any>('/microsoft/apps/studios/labels/sample/random', {
      method: 'POST',
      body: JSON.stringify({ task_id, sample_id })
    }),
    submitSample: (data: { task_id: string; labeler_id: string; sample_id: string; label: any; comment: string }) => fetchApi<any>('/microsoft/apps/studios/labels/sample/submit', {
      method: 'POST',
      body: JSON.stringify(data)
    }),
    export: (data: { task_id: string; labeler_ids?: string[] | null; include_unfinished?: boolean }) => fetchApi<any>('/microsoft/apps/studios/labels/export', {
      method: 'POST',
      body: JSON.stringify(data)
    }),
    delete: (id: string) => fetchApi<any>('/microsoft/apps/studios/labels/delete', {
      method: 'POST',
      body: JSON.stringify({ id })
    })
  },

  // Reports
  reports: {
    list: () => fetchApi<any[]>('/microsoft/apps/studios/reports'),
    getDetails: (id: string) => fetchApi<any>('/microsoft/apps/studios/reports/get', {
      method: 'POST',
      body: JSON.stringify({ id })
    }),
    update: (data: { id: string; content?: string; name?: string; description?: string; extra?: any }) => 
      fetchApi<any>('/microsoft/apps/studios/reports/update', {
        method: 'POST',
        body: JSON.stringify(data)
      }),
    delete: (id: string) => 
      fetchApi<any>('/microsoft/apps/studios/reports/delete', {
        method: 'POST',
        body: JSON.stringify({ id })
      })
  }
};