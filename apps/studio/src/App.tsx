import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/chat/ChatArea';
import { DatasetView } from './components/views/DatasetView';
import { JobView } from './components/views/JobView';
import { LabelView } from './components/views/LabelView';
import { ReportView } from './components/views/ReportView';
import { LoginView } from './components/views/LoginView';
import { useStore } from './store/useStore';
import { Menu } from 'lucide-react';

function App() {
  const { currentView, isAuthenticated, sessions, activeSessionId, selectedEntityId, toggleSidebar } = useStore();
  const activeSession = sessions.find(s => s.id === activeSessionId);

  if (!isAuthenticated) {
    return <LoginView />;
  }

  return (
    <div className="flex h-screen w-full bg-background text-foreground overflow-hidden">
      <Sidebar />
      
      <main className="flex-1 flex flex-col min-w-0">
        <header className="h-14 border-b border-border flex items-center px-4 shrink-0 bg-background/50 backdrop-blur-sm gap-3">
          <button 
            onClick={toggleSidebar}
            className="p-2 -ml-2 hover:bg-secondary rounded-md text-muted-foreground transition-colors"
            title="Toggle Sidebar"
          >
            <Menu size={18} />
          </button>
          <h1 className="font-semibold text-sm">
            {currentView === 'chat' 
              ? (activeSession?.title || 'Active Session') 
              : <span className="capitalize">{currentView} {selectedEntityId ? 'Details' : 'Overview'}</span>}
          </h1>
        </header>
        
        <div className="flex-1 overflow-hidden relative">
          {currentView === 'chat' && <ChatArea />}
          {currentView === 'dataset' && <DatasetView />}
          {currentView === 'job' && <JobView />}
          {currentView === 'label' && <LabelView />}
          {currentView === 'report' && <ReportView />}
        </div>
      </main>
    </div>
  );
}

export default App;