import { useEffect, useState, useRef } from 'react';
import { useStore } from '../../store/useStore';
import { TerminalSquare, StopCircle, RefreshCw, Search, Share2, Check } from 'lucide-react';

export function JobView() {
  const { jobs, selectedEntityId, setView } = useStore();
  const [search, setSearch] = useState('');
  const [copied, setCopied] = useState(false);
  
  const job = jobs.find(j => j.id === selectedEntityId);
  const [logs, setLogs] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!job) return;
    
    const initialLogs = [
      `[${new Date().toISOString()}] INITIALIZING JOB: ${job.name}`,
      `[${new Date().toISOString()}] Setting up container environment...`,
      `[${new Date().toISOString()}] Downloading base weights...`,
    ];
    setLogs(initialLogs);

    if (job.status === 'completed') {
      setLogs(prev => [...prev, `[${new Date().toISOString()}] Job completed successfully. Weights saved to output/`]);
      return;
    }

    const interval = setInterval(() => {
      setLogs(prev => [...prev, `[${new Date().toISOString()}] Processing batch ${Math.floor(Math.random() * 1000)}... loss: ${(Math.random() * 2).toFixed(4)}`]);
      
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [job]);

  const handleShare = () => {
    navigator.clipboard.writeText(`${window.location.origin}/jobs/${job?.id}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!selectedEntityId) {
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
              </div>
              <div className="flex items-center gap-2 mb-3">
                <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider ${
                  j.status === 'running' ? 'bg-blue-500/10 text-blue-600' : 'bg-green-500/10 text-green-600'
                }`}>
                  {j.status}
                </span>
              </div>
              <div className="flex justify-between text-xs mb-1 text-muted-foreground">
                <span>Progress</span>
                <span>{j.progress}%</span>
              </div>
              <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                <div 
                  className={`h-full transition-all ${j.status === 'running' ? 'bg-blue-500' : 'bg-green-500'}`} 
                  style={{ width: `${j.progress}%` }} 
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

  if (!job) return <div className="p-8 text-muted-foreground">Job not found</div>;

  return (
    <div className="h-full flex flex-col bg-background p-8">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-3 mb-2">
            <TerminalSquare className={job.status === 'running' ? 'text-blue-500' : 'text-green-500'} /> 
            {job.name}
          </h2>
          <div className="flex items-center gap-3 text-sm">
            <span className={`px-2 py-1 rounded-md text-xs uppercase font-bold tracking-wider ${
              job.status === 'running' ? 'bg-blue-500/10 text-blue-600' : 'bg-green-500/10 text-green-600'
            }`}>
              {job.status}
            </span>
            <span className="text-muted-foreground">ID: {job.id}</span>
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <button 
            onClick={handleShare}
            className="px-4 py-2 flex items-center gap-2 bg-card border border-border hover:bg-secondary rounded-md text-sm font-medium transition-colors"
          >
            {copied ? <Check size={16} className="text-green-500" /> : <Share2 size={16} className="text-blue-500" />}
            <span className="hidden sm:inline">{copied ? 'Copied Link' : 'Share Job'}</span>
          </button>
          <button className="px-4 py-2 flex items-center gap-2 bg-secondary hover:bg-secondary/80 rounded-md text-sm font-medium transition-colors">
            <RefreshCw size={16} /> Restart
          </button>
          {job.status === 'running' && (
            <button className="px-4 py-2 flex items-center gap-2 bg-destructive/10 hover:bg-destructive/20 text-destructive rounded-md text-sm font-medium transition-colors">
              <StopCircle size={16} /> Cancel Job
            </button>
          )}
        </div>
      </div>

      <div className="bg-card border border-border rounded-xl p-6 mb-6">
        <div className="flex justify-between text-sm mb-2">
          <span className="font-medium">Overall Progress</span>
          <span className="font-mono">{job.progress}%</span>
        </div>
        <div className="h-3 w-full bg-secondary rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-500 ${job.status === 'running' ? 'bg-blue-500' : 'bg-green-500'}`} 
            style={{ width: `${job.progress}%` }} 
          />
        </div>
      </div>

      <div className="flex-1 bg-slate-950 border border-border rounded-xl flex flex-col overflow-hidden shadow-inner">
        <div className="px-4 py-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-sm text-slate-400">
          <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
          <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
          <span className="ml-2 font-mono text-xs">Terminal Logs</span>
        </div>
        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-4 font-mono text-sm text-slate-300 space-y-1"
        >
          {logs.map((log, i) => (
            <div key={i} className="hover:bg-white/5 px-1 rounded">
              {log}
            </div>
          ))}
          {job.status === 'running' && (
            <div className="animate-pulse mt-2 text-blue-400">_</div>
          )}
        </div>
      </div>
    </div>
  );
}