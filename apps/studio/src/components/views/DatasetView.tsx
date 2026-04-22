import { useState } from 'react';
import { useStore } from '../../store/useStore';
import { Database, BarChart2, Table, Search, Share2, Check } from 'lucide-react';

export function DatasetView() {
  const { datasets, selectedEntityId, setView } = useStore();
  const [search, setSearch] = useState('');
  const [copied, setCopied] = useState(false);
  
  const dataset = datasets.find(d => d.id === selectedEntityId);

  const handleShare = () => {
    navigator.clipboard.writeText(`${window.location.origin}/datasets/${dataset?.id}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!selectedEntityId) {
    const filtered = datasets.filter(d => d.name.toLowerCase().includes(search.toLowerCase()));
    return (
      <div className="h-full flex flex-col bg-background p-8">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold flex items-center gap-3">
            <Database className="text-blue-400" /> Datasets Overview
          </h2>
          <div className="relative w-64">
            <Search className="absolute left-3 top-2.5 text-muted-foreground" size={16} />
            <input 
              type="text" 
              placeholder="Search datasets..." 
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-secondary/30 border border-border rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 overflow-y-auto">
          {filtered.map(d => (
            <div 
              key={d.id} 
              onClick={() => setView('dataset', d.id)}
              className="p-5 bg-card border border-border rounded-xl cursor-pointer hover:border-blue-500/50 transition-colors group"
            >
              <div className="flex items-start justify-between mb-3">
                <h3 className="font-semibold group-hover:text-blue-400 transition-colors truncate pr-2">{d.name}</h3>
                <Database size={16} className="text-muted-foreground" />
              </div>
              <p className="text-sm text-muted-foreground">Rows: {d.rows.toLocaleString()}</p>
              <p className="text-sm text-muted-foreground">Size: {d.size}</p>
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="col-span-full py-12 text-center text-muted-foreground">No datasets found matching "{search}"</div>
          )}
        </div>
      </div>
    );
  }

  if (!dataset) return <div className="p-8 text-muted-foreground">Dataset not found</div>;

  const mockRows = Array.from({ length: 15 }).map((_, i) => ({
    id: i + 1,
    text: `Sample review text ${i + 1} for model training...`,
    label: i % 3 === 0 ? 'positive' : i % 2 === 0 ? 'neutral' : 'negative',
    confidence: (Math.random() * 0.5 + 0.5).toFixed(2)
  }));

  return (
    <div className="h-full flex flex-col bg-background overflow-y-auto p-8 space-y-8">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-3 mb-2">
            <Database className="text-blue-400" /> {dataset.name}
          </h2>
          <p className="text-muted-foreground">ID: {dataset.id} • {dataset.rows.toLocaleString()} rows • {dataset.size}</p>
        </div>
        <button 
          onClick={handleShare}
          className="flex items-center gap-2 px-3 py-1.5 bg-card border border-border hover:bg-secondary rounded-lg text-sm font-medium transition-colors"
        >
          {copied ? <Check size={16} className="text-green-500" /> : <Share2 size={16} className="text-blue-400" />}
          <span className="hidden sm:inline">{copied ? 'Copied Link' : 'Share Dataset'}</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-card border border-border rounded-xl p-6">
          <h3 className="font-semibold mb-4 flex items-center gap-2"><BarChart2 size={18}/> Label Distribution</h3>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1"><span>Positive</span><span>45%</span></div>
              <div className="h-2 bg-secondary rounded-full"><div className="h-full bg-green-500 rounded-full w-[45%]"></div></div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1"><span>Neutral</span><span>35%</span></div>
              <div className="h-2 bg-secondary rounded-full"><div className="h-full bg-yellow-500 rounded-full w-[35%]"></div></div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1"><span>Negative</span><span>20%</span></div>
              <div className="h-2 bg-secondary rounded-full"><div className="h-full bg-red-500 rounded-full w-[20%]"></div></div>
            </div>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl p-6">
          <h3 className="font-semibold mb-4">Dataset Splits</h3>
          <div className="flex items-center gap-4 h-full pb-4">
            <div className="flex-1 flex flex-col justify-center items-center p-4 bg-secondary/50 rounded-lg">
              <span className="text-2xl font-bold text-blue-400">80%</span>
              <span className="text-xs text-muted-foreground uppercase tracking-wider">Train</span>
            </div>
            <div className="flex-1 flex flex-col justify-center items-center p-4 bg-secondary/50 rounded-lg">
              <span className="text-2xl font-bold text-purple-400">10%</span>
              <span className="text-xs text-muted-foreground uppercase tracking-wider">Valid</span>
            </div>
            <div className="flex-1 flex flex-col justify-center items-center p-4 bg-secondary/50 rounded-lg">
              <span className="text-2xl font-bold text-green-400">10%</span>
              <span className="text-xs text-muted-foreground uppercase tracking-wider">Test</span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-card border border-border rounded-xl flex flex-col flex-1 overflow-hidden min-h-[400px]">
        <div className="p-4 border-b border-border font-semibold flex items-center gap-2">
          <Table size={18}/> Data Preview (First 15 rows)
        </div>
        <div className="overflow-x-auto flex-1">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-muted-foreground uppercase bg-secondary/30 sticky top-0">
              <tr>
                <th className="px-6 py-3 border-b border-border">ID</th>
                <th className="px-6 py-3 border-b border-border">Text</th>
                <th className="px-6 py-3 border-b border-border">Label</th>
                <th className="px-6 py-3 border-b border-border">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {mockRows.map(row => (
                <tr key={row.id} className="border-b border-border hover:bg-secondary/20">
                  <td className="px-6 py-3 font-mono text-xs">{row.id}</td>
                  <td className="px-6 py-3 max-w-md truncate">{row.text}</td>
                  <td className="px-6 py-3">
                    <span className="px-2 py-1 bg-secondary rounded-md text-xs">{row.label}</span>
                  </td>
                  <td className="px-6 py-3 font-mono">{row.confidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}