import { Database, TerminalSquare, MessageSquare, Settings, LogOut, Tag, FileText, Plus, X, Trash2 } from 'lucide-react';
import { useStore } from '../store/useStore';
import { cn } from '../lib/utils';

export function Sidebar() {
  const { 
    datasets, jobs, reports, labelTasks, 
    currentView, setView, logout,
    sessions, activeSessionId, createNewSession, setActiveSession, deleteSession,
    isSidebarOpen, toggleSidebar,
    selectedDatasetId, selectedJobId, selectedLabelId, selectedReportId
  } = useStore();

  return (
    <>
      {/* Mobile Backdrop */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden" 
          onClick={toggleSidebar}
        />
      )}

      <div className={cn(
        "fixed lg:static inset-y-0 left-0 z-50 h-full bg-background lg:bg-secondary/30 transition-all duration-300 flex flex-col overflow-hidden shrink-0 shadow-2xl lg:shadow-none",
        isSidebarOpen 
          ? "w-[260px] translate-x-0 border-r border-border" 
          : "w-[260px] -translate-x-full lg:translate-x-0 lg:w-[68px] border-r border-border"
      )}>
        
        {/* Header */}
        <div className={cn("h-14 p-4 flex items-center shrink-0 border-b border-border/50", isSidebarOpen ? "justify-between" : "justify-between lg:justify-center")}>
          <div className="flex items-center gap-2 overflow-hidden">
            <div className="w-6 h-6 rounded bg-primary flex items-center justify-center text-primary-foreground shrink-0">
              <TerminalSquare size={14} />
            </div>
            <span className={cn("font-bold text-lg whitespace-nowrap", isSidebarOpen ? "block" : "lg:hidden")}>
              Ads Studio
            </span>
          </div>
          <button 
            onClick={toggleSidebar}
            className={cn("lg:hidden p-1.5 hover:bg-secondary rounded-md text-muted-foreground transition-colors", isSidebarOpen ? "block" : "hidden")}
          >
            <X size={18} />
          </button>
        </div>

        {/* Navigation Area */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden p-3 space-y-6">
          
          {/* Sessions */}
          <div>
            <div className={cn("flex items-center mb-2 px-2", isSidebarOpen ? "justify-between" : "justify-between lg:justify-center lg:px-0")}>
              <div className={cn("text-xs font-semibold text-muted-foreground uppercase tracking-wider", isSidebarOpen ? "block" : "lg:hidden")}>Sessions</div>
              <button 
                onClick={createNewSession}
                className="w-5 h-5 flex items-center justify-center rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                title="New Chat"
              >
                <Plus size={14} />
              </button>
            </div>
            {sessions.map((session) => (
              <div key={session.id} className="relative group w-full flex items-center">
                <button
                  title={!isSidebarOpen ? session.title : undefined}
                  onClick={() => setActiveSession(session.id)}
                  className={cn(
                    "w-full flex items-center gap-2 py-2 text-sm rounded-md transition-colors pr-8",
                    isSidebarOpen ? "px-2" : "px-2 lg:px-0 lg:justify-center",
                    currentView === 'chat' && activeSessionId === session.id 
                      ? "bg-secondary text-foreground" 
                      : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
                  )}
                >
                  <MessageSquare size={16} className="shrink-0" />
                  <span className={cn("truncate text-left flex-1", isSidebarOpen ? "block" : "lg:hidden")}>{session.title}</span>
                </button>
                {isSidebarOpen && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteSession(session.id);
                    }}
                    className="absolute right-2 opacity-0 group-hover:opacity-100 p-1 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded transition-all"
                    title="Delete Session"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Datasets */}
          <div>
            <div 
              className={cn("text-xs font-semibold text-muted-foreground mb-2 px-2 uppercase tracking-wider cursor-pointer hover:text-foreground transition-colors", isSidebarOpen ? "block" : "lg:hidden")}
              onClick={() => setView('dataset', null)}
            >
              Datasets
            </div>
            {datasets.map((dataset) => (
              <button
                key={dataset.id}
                title={!isSidebarOpen ? dataset.name : undefined}
                onClick={() => setView('dataset', dataset.id)}
                className={cn(
                  "w-full flex items-center gap-2 py-2 text-sm rounded-md transition-colors",
                  isSidebarOpen ? "px-2" : "px-2 lg:px-0 lg:justify-center",
                  currentView === 'dataset' && selectedDatasetId === dataset.id ? "bg-secondary text-foreground" : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
                )}
              >
                <Database size={16} className="shrink-0" />
                <span className={cn("truncate text-left flex-1", isSidebarOpen ? "block" : "lg:hidden")}>{dataset.name}</span>
              </button>
            ))}
          </div>

          {/* Jobs */}
          <div>
            <div 
              className={cn("text-xs font-semibold text-muted-foreground mb-2 px-2 uppercase tracking-wider cursor-pointer hover:text-foreground transition-colors", isSidebarOpen ? "block" : "lg:hidden")}
              onClick={() => setView('job', null)}
            >
              Jobs
            </div>
            {jobs.map((job) => (
              <button
                key={job.id}
                title={!isSidebarOpen ? job.name : undefined}
                onClick={() => setView('job', job.id)}
                className={cn(
                  "w-full flex items-center gap-2 py-2 text-sm rounded-md transition-colors",
                  isSidebarOpen ? "px-2" : "px-2 lg:px-0 lg:justify-center",
                  currentView === 'job' && selectedJobId === job.id ? "bg-secondary text-foreground" : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
                )}
              >
                <TerminalSquare size={16} className="shrink-0" />
                <div className={cn("flex-1 text-left truncate", isSidebarOpen ? "block" : "lg:hidden")}>{job.name}</div>
                {job.status === 'running' && <div className={cn("w-2 h-2 rounded-full bg-blue-500 animate-pulse shrink-0", isSidebarOpen ? "block" : "lg:hidden")} />}
                {job.status === 'completed' && <div className={cn("w-2 h-2 rounded-full bg-green-500 shrink-0", isSidebarOpen ? "block" : "lg:hidden")} />}
              </button>
            ))}
          </div>

          {/* Labels */}
          <div>
            <div 
              className={cn("text-xs font-semibold text-muted-foreground mb-2 px-2 uppercase tracking-wider cursor-pointer hover:text-foreground transition-colors", isSidebarOpen ? "block" : "lg:hidden")}
              onClick={() => setView('label', null)}
            >
              Labels
            </div>
            {labelTasks.map((task) => (
              <button
                key={task.id}
                title={!isSidebarOpen ? task.name : undefined}
                onClick={() => setView('label', task.id)}
                className={cn(
                  "w-full flex items-center gap-2 py-2 text-sm rounded-md transition-colors",
                  isSidebarOpen ? "px-2" : "px-2 lg:px-0 lg:justify-center",
                  currentView === 'label' && selectedLabelId === task.id ? "bg-secondary text-foreground" : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
                )}
              >
                <Tag size={16} className="shrink-0" />
                <span className={cn("truncate text-left flex-1", isSidebarOpen ? "block" : "lg:hidden")}>{task.name}</span>
              </button>
            ))}
          </div>

          {/* Reports */}
          <div>
            <div 
              className={cn("text-xs font-semibold text-muted-foreground mb-2 px-2 uppercase tracking-wider cursor-pointer hover:text-foreground transition-colors", isSidebarOpen ? "block" : "lg:hidden")}
              onClick={() => setView('report', null)}
            >
              Reports
            </div>
            {reports.map((report) => (
              <button
                key={report.id}
                title={!isSidebarOpen ? report.name : undefined}
                onClick={() => setView('report', report.id)}
                className={cn(
                  "w-full flex items-center gap-2 py-2 text-sm rounded-md transition-colors",
                  isSidebarOpen ? "px-2" : "px-2 lg:px-0 lg:justify-center",
                  currentView === 'report' && selectedReportId === report.id ? "bg-secondary text-foreground" : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
                )}
              >
                <FileText size={16} className="shrink-0" />
                <span className={cn("truncate text-left flex-1", isSidebarOpen ? "block" : "lg:hidden")}>{report.name}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-3 border-t border-border/50 space-y-2 shrink-0">
          <button 
            title={!isSidebarOpen ? "Settings" : undefined}
            className={cn("w-full flex items-center gap-2 py-2 text-sm text-muted-foreground hover:bg-secondary rounded-md transition-colors", isSidebarOpen ? "px-2" : "px-2 lg:px-0 lg:justify-center")}
          >
            <Settings size={16} className="shrink-0" />
            <span className={cn("truncate", isSidebarOpen ? "block" : "lg:hidden")}>Settings</span>
          </button>
          <button 
            onClick={logout}
            title={!isSidebarOpen ? "Sign Out" : undefined}
            className={cn("w-full flex items-center gap-2 py-2 text-sm text-destructive hover:bg-destructive/10 rounded-md transition-colors", isSidebarOpen ? "px-2" : "px-2 lg:px-0 lg:justify-center")}
          >
            <LogOut size={16} className="shrink-0" />
            <span className={cn("truncate", isSidebarOpen ? "block" : "lg:hidden")}>Sign Out</span>
          </button>
        </div>

      </div>
    </>
  );
}