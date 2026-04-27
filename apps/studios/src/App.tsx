import { useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/chat/ChatArea';
import { DatasetView } from './components/views/DatasetView';
import { JobView } from './components/views/JobView';
import { LabelView } from './components/views/LabelView';
import { ReportView } from './components/views/ReportView';
import { LoginView } from './components/views/LoginView';
import { SettingsView } from './components/views/SettingsView';
import { useStore } from './store/useStore';
import { Menu } from 'lucide-react';
import { cn } from './lib/utils';

function App() {
  const { 
    currentView, isAuthenticated, sessions, activeSessionId, 
    selectedDatasetId, selectedJobId, selectedLabelId, selectedReportId, toggleSidebar,
    initData
  } = useStore();
  const activeSession = sessions.find(s => s.id === activeSessionId);

  // 当用户已登录时，确保初始化基础数据（包含模型列表等）
  useEffect(() => {
    if (isAuthenticated) {
      initData();
    }
  }, [isAuthenticated, initData]);

  if (!isAuthenticated) {
    return <LoginView />;
  }

  const getActiveId = () => {
    if (currentView === 'dataset') return selectedDatasetId;
    if (currentView === 'job') return selectedJobId;
    if (currentView === 'label') return selectedLabelId;
    if (currentView === 'report') return selectedReportId;
    return null;
  };

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
              : <span className="capitalize">{currentView} {getActiveId() ? 'Details' : 'Overview'}</span>}
          </h1>
        </header>
        
        <div className="flex-1 overflow-hidden relative">
          <div className={cn("absolute inset-0", currentView !== 'chat' && "hidden")}>
            <ChatArea />
          </div>
          <div className={cn("absolute inset-0", currentView !== 'dataset' && "hidden")}>
            <DatasetView />
          </div>
          <div className={cn("absolute inset-0", currentView !== 'job' && "hidden")}>
            <JobView />
          </div>
          <div className={cn("absolute inset-0", currentView !== 'label' && "hidden")}>
            <LabelView />
          </div>
          <div className={cn("absolute inset-0", currentView !== 'report' && "hidden")}>
            <ReportView />
          </div>
          <div className={cn("absolute inset-0", currentView !== 'settings' && "hidden")}>
            <SettingsView />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;