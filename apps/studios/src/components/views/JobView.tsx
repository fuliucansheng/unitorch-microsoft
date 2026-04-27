import { useEffect, useState, useRef } from 'react';
import { useStore } from '../../store/useStore';
import { api } from '../../lib/api';
import { TerminalSquare, StopCircle, RefreshCw, Search, Share2, Check, Settings2, Info, Trash2 } from 'lucide-react';

export function JobView() {
  const { jobs, selectedJobId, setView, initData } = useStore();
  const [search, setSearch] = useState('');
  const [copied, setCopied] = useState(false);
  
  const baseJob = jobs.find(j => j.id === selectedJobId);
  const [jobDetails, setJobDetails] = useState<any>(null);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!selectedJobId) return;
    
    let active = true;
    
    const fetchDetailsAndLogs = async () => {
      try {
        const [details, logsRes] = await Promise.all([
          api.jobs.getDetails(selectedJobId).catch(err => {
            console.error("Failed to fetch job details:", err);
            return null;
          }),
          api.jobs.getLogs(selectedJobId, 100).catch(err => {
            console.error("Failed to fetch job logs:", err);
            return null;
          })
        ]);
        
        if (active) {
          if (details) setJobDetails(details);
          if (logsRes?.lines) setJobLogs(logsRes.lines);
        }
      } catch (error) {
        console.error("Failed during fetchDetailsAndLogs:", error);
      }
    };

    fetchDetailsAndLogs();
    
    // Poll every 3 seconds if the job is running
    const interval = setInterval(() => {
      if (jobDetails?.status === 'running' || baseJob?.status === 'running') {
        fetchDetailsAndLogs();
      }
    }, 3000);

    return () => {
      active = false;
      clearInterval(interval);
      setJobLogs([]); // Reset logs when changing jobs
    };
  }, [selectedJobId, baseJob?.status, jobDetails?.status]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [jobLogs]);

  const handleShare = () => {
    navigator.clipboard.writeText(`${window.location.origin}/jobs/${selectedJobId}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCancel = async () => {
    if (!selectedJobId) return;
    try {
      const res = await api.jobs.cancel(selectedJobId);
      if (res) setJobDetails(res);
      await initData();
    } catch (e) {
      console.error(e);
    }
  };

  const handleRestart = async () => {
    if (!selectedJobId) return;
    try {
      const res = await api.jobs.restart(selectedJobId);
      if (res) setJobDetails(res);
      await initData();
    } catch (e) {
      console.error(e);
    }
  };

  const handleDelete = async () => {
    if (!selectedJobId) return;
    const currentJob = { ...baseJob, ...jobDetails };
    
    if (currentJob.status === 'running') {
      alert('Cannot delete a running job. Please cancel it first.');
      return;
    }
    
    if (!confirm('Are you sure you want to delete this job? This action cannot be undone.')) return;
    
    try {
      await api.jobs.delete(selectedJobId);
      await initData();
      setView('job', null);
    } catch (err) {
      console.error('Failed to delete job:', err);
      alert('Failed to delete job');
    }
  };

  if (!selectedJobId) {
    const filtered = jobs.filter(j => j.name.toLowerCase().includes(search.toLowerCase()));
    return (
      <div className="h-full flex flex-col bg-background p-8">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold flex items-center gap-3">
            <TerminalSquare className="text-primary" /> Jobs Overview
          </h2>
          <div className="relative w-64">
            <Search className="absolute left-3 top-2.5 text-muted-foreground" size={16} />
            <input 
              type="text" 
              placeholder="Search jobs..." 
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-secondary/30 border border-border rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 overflow-y-auto">
          {filtered.map(j => (
            <div 
              key={j.id} 
              onClick={() => setView('job', j.id)}
              className="p-5 bg-card border border-border rounded-xl cursor-pointer hover:border-primary/50 transition-colors group"
            >
              <div className="flex items-start justify-between mb-3">
                <h3 className="font-semibold group-hover:text-primary transition-colors truncate pr-2">{j.name}</h3>
                {j.status === 'running' && <div className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse mt-1" />}
                {j.status === 'completed' && <div className="w-2.5 h-2.5 rounded-full bg-green-500 mt-1" />}
                {j.status === 'cancelled' && <div className="w-2.5 h-2.5 rounded-full bg-gray-500 mt-1" />}
              </div>
              <div className="flex items-center gap-2 mb-3">
                <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider ${
                  j.status === 'running' ? 'bg-blue-500/10 text-blue-600' : 
                  j.status === 'completed' ? 'bg-green-500/10 text-green-600' : 'bg-gray-500/10 text-gray-600'
                }`}>
                  {j.status}
                </span>
              </div>
              <div className="flex justify-between text-xs mb-1 text-muted-foreground">
                <span>Progress</span>
                <span>{j.progress ?? 100}%</span>
              </div>
              <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                <div 
                  className={`h-full transition-all ${
                    j.status === 'running' ? 'bg-blue-500' : 
                    j.status === 'completed' ? 'bg-green-500' : 'bg-gray-500'
                  }`} 
                  style={{ width: `${j.progress ?? 100}%` }} 
                />
              </div>
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="col-span-full py-12 text-center text-muted-foreground">No jobs found matching "{search}"</div>
          )}
        </div>
      </div>
    );
  }

  const currentJob = { ...baseJob, ...jobDetails };
  if (!currentJob) return <div className="p-8 text-muted-foreground">Job not found</div>;

  const displayLogs = jobLogs.length > 0 
    ? jobLogs 
    : (currentJob.status === 'running' ? ['Waiting for logs...'] : ['No logs available.']);
    
  const progress = currentJob.progress ?? (currentJob.status === 'running' ? 50 : 100);

  return (
    <div className="h-full flex flex-col bg-background p-8 overflow-y-auto">
      <div className="flex items-start justify-between mb-6 shrink-0">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-3 mb-2">
            <TerminalSquare className={
              currentJob.status === 'running' ? 'text-blue-500' : 
              currentJob.status === 'completed' ? 'text-green-500' : 'text-gray-500'
            } /> 
            {currentJob.name}
          </h2>
          <div className="flex items-center gap-3 text-sm">
            <span className={`px-2 py-1 rounded-md text-xs uppercase font-bold tracking-wider ${
              currentJob.status === 'running' ? 'bg-blue-500/10 text-blue-600' : 
              currentJob.status === 'completed' ? 'bg-green-500/10 text-green-600' : 'bg-gray-500/10 text-gray-600'
            }`}>
              {currentJob.status}
            </span>
            <span className="text-muted-foreground">ID: {currentJob.id}</span>
          </div>
          {currentJob.description && (
            <p className="text-sm text-muted-foreground mt-2">{currentJob.description}</p>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button 
            title="Share Job"
            onClick={handleShare}
            className="p-2 bg-card border border-border hover:bg-secondary rounded-md transition-colors"
          >
            {copied ? <Check size={18} className="text-green-500" /> : <Share2 size={18} className="text-blue-500" />}
          </button>
          <button 
            title="Restart"
            onClick={handleRestart}
            className="p-2 bg-secondary hover:bg-secondary/80 rounded-md transition-colors"
          >
            <RefreshCw size={18} />
          </button>
          {currentJob.status === 'running' && (
            <button 
              title="Cancel Job"
              onClick={handleCancel}
              className="p-2 bg-destructive/10 hover:bg-destructive/20 text-destructive rounded-md transition-colors"
            >
              <StopCircle size={18} />
            </button>
          )}
          <button 
            title="Delete Job"
            onClick={handleDelete}
            className="p-2 bg-destructive/10 hover:bg-destructive/20 text-destructive rounded-md transition-colors"
          >
            <Trash2 size={18} />
          </button>
        </div>
      </div>

      {jobDetails && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 shrink-0">
          <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
            <h3 className="font-semibold text-sm mb-4 flex items-center gap-2"><Info size={16} className="text-muted-foreground"/> Execution Details</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center"><span className="text-muted-foreground">Type</span><span className="px-2 py-0.5 bg-secondary rounded text-xs capitalize">{currentJob.type || 'Standard'}</span></div>
              <div className="flex justify-between items-center"><span className="text-muted-foreground">PID</span><span className="font-mono text-xs">{currentJob.pid || '-'}</span></div>
              <div className="flex justify-between items-center"><span className="text-muted-foreground">Exit Code</span><span className="font-mono text-xs">{currentJob.exit_code ?? '-'}</span></div>
              <div className="flex justify-between items-center"><span className="text-muted-foreground">Started</span><span className="text-xs">{currentJob.started_at ? new Date(currentJob.started_at).toLocaleString() : '-'}</span></div>
              <div className="flex justify-between items-center"><span className="text-muted-foreground">Finished</span><span className="text-xs">{currentJob.finished_at ? new Date(currentJob.finished_at).toLocaleString() : '-'}</span></div>
            </div>
          </div>
          <div className="bg-card border border-border rounded-xl p-5 shadow-sm flex flex-col">
            <h3 className="font-semibold text-sm mb-4 flex items-center gap-2"><Settings2 size={16} className="text-muted-foreground"/> Environment & Command</h3>
            <div className="space-y-4 text-sm flex-1">
              <div>
                <span className="text-muted-foreground block mb-1.5 text-xs uppercase tracking-wider">Command</span>
                <code className="bg-secondary px-3 py-2 rounded-lg block text-xs overflow-x-auto whitespace-nowrap text-foreground font-mono">
                  {currentJob.command || 'N/A'}
                </code>
              </div>
              <div>
                <span className="text-muted-foreground block mb-1.5 text-xs uppercase tracking-wider">Working Directory</span>
                <code className="bg-secondary px-3 py-2 rounded-lg block text-xs overflow-x-auto whitespace-nowrap text-foreground font-mono">
                  {currentJob.workdir || 'N/A'}
                </code>
              </div>
              {currentJob.dataset_ids && currentJob.dataset_ids.length > 0 && (
                <div>
                  <span className="text-muted-foreground block mb-1.5 text-xs uppercase tracking-wider">Linked Datasets</span>
                  <div className="flex flex-wrap gap-1.5">
                    {currentJob.dataset_ids.map((id: string) => (
                      <span key={id} className="px-2 py-1 bg-secondary border border-border text-[10px] rounded-md font-mono">{id.substring(0, 8)}...</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="bg-card border border-border rounded-xl p-6 mb-6 shrink-0 shadow-sm">
        <div className="flex justify-between text-sm mb-2">
          <span className="font-medium">Overall Progress</span>
          <span className="font-mono">{progress}%</span>
        </div>
        <div className="h-3 w-full bg-secondary rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-500 ${
              currentJob.status === 'running' ? 'bg-blue-500' : 
              currentJob.status === 'completed' ? 'bg-green-500' : 'bg-gray-500'
            }`} 
            style={{ width: `${progress}%` }} 
          />
        </div>
      </div>

      <div className="flex-1 bg-slate-950 border border-border rounded-xl flex flex-col overflow-hidden shadow-inner min-h-[300px]">
        <div className="px-4 py-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-sm text-slate-400 shrink-0">
          <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
          <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
          <span className="ml-2 font-mono text-xs">Terminal Logs</span>
        </div>
        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-4 font-mono text-sm text-slate-300 space-y-1 whitespace-pre-wrap"
        >
          {displayLogs.map((log: string, i: number) => (
            <div key={i} className="hover:bg-white/5 px-1 rounded break-words">
              {log}
            </div>
          ))}
          {currentJob.status === 'running' && (
            <div className="animate-pulse mt-2 text-blue-400">_</div>
          )}
        </div>
      </div>
    </div>
  );
}