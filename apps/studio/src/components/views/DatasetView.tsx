import { useState, useEffect } from 'react';
import { useStore } from '../../store/useStore';
import { api } from '../../lib/api';
import { Database, BarChart2, Table, Search, Share2, Check, Loader2 } from 'lucide-react';

export function DatasetView() {
  const { datasets, selectedDatasetId, setView } = useStore();
  const [search, setSearch] = useState('');
  const [copied, setCopied] = useState(false);
  
  const baseDataset = datasets.find(d => d.id === selectedDatasetId);
  const [details, setDetails] = useState<any>(null);
  const [preview, setPreview] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedDatasetId) return;
    
    setLoading(true);
    Promise.all([
      api.datasets.getDetails(selectedDatasetId).catch(console.error),
      api.datasets.getPreview(selectedDatasetId).catch(console.error)
    ]).then(([detailsData, previewData]) => {
      if (detailsData) setDetails(detailsData);
      if (previewData) setPreview(previewData);
      setLoading(false);
    });
  }, [selectedDatasetId]);

  const handleShare = () => {
    navigator.clipboard.writeText(`${window.location.origin}/datasets/${selectedDatasetId}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!selectedDatasetId) {
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
              <p className="text-sm text-muted-foreground">Rows: {d.rows ? d.rows.toLocaleString() : 'N/A'}</p>
              <p className="text-sm text-muted-foreground">Size: {d.size || 'Unknown'}</p>
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="col-span-full py-12 text-center text-muted-foreground">No datasets found matching "{search}"</div>
          )}
        </div>
      </div>
    );
  }

  const dataset = details || baseDataset;
  if (!dataset) return <div className="p-8 text-muted-foreground">Dataset not found</div>;

  return (
    <div className="h-full flex flex-col bg-background overflow-y-auto p-8 space-y-8">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-3 mb-2">
            <Database className="text-blue-400" /> {dataset.name}
          </h2>
          <p className="text-muted-foreground">
            ID: {dataset.id} • {dataset.rows ? dataset.rows.toLocaleString() : 0} rows • {dataset.size || 'Unknown'}
          </p>
          {dataset.description && (
            <p className="text-sm mt-2 text-foreground/80">{dataset.description}</p>
          )}
        </div>
        <button 
          onClick={handleShare}
          className="flex items-center gap-2 px-3 py-1.5 bg-card border border-border hover:bg-secondary rounded-lg text-sm font-medium transition-colors"
        >
          {copied ? <Check size={16} className="text-green-500" /> : <Share2 size={16} className="text-blue-400" />}
          <span className="hidden sm:inline">{copied ? 'Copied Link' : 'Share Dataset'}</span>
        </button>
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center text-muted-foreground">
          <Loader2 className="animate-spin mr-2" size={24} /> Loading dataset details...
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Label Distribution */}
            <div className="bg-card border border-border rounded-xl p-6">
              <h3 className="font-semibold mb-4 flex items-center gap-2"><BarChart2 size={18}/> Label Distribution</h3>
              <div className="space-y-3">
                {details?.details?.labels_distributions ? (
                  Object.entries(details.details.labels_distributions).map(([label, percentage]: [string, any], idx) => {
                    const colors = ['bg-green-500', 'bg-blue-500', 'bg-purple-500', 'bg-yellow-500', 'bg-red-500'];
                    const color = colors[idx % colors.length];
                    return (
                      <div key={label}>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="capitalize">{label}</span>
                          <span>{percentage}%</span>
                        </div>
                        <div className="h-2 bg-secondary rounded-full">
                          <div className={`h-full ${color} rounded-full`} style={{ width: `${percentage}%` }}></div>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <p className="text-sm text-muted-foreground">No distribution data available.</p>
                )}
              </div>
            </div>

            {/* Dataset Splits */}
            <div className="bg-card border border-border rounded-xl p-6">
              <h3 className="font-semibold mb-4">Dataset Splits</h3>
              <div className="flex items-center gap-4 h-full pb-4">
                {details?.details?.splits ? (
                  Object.entries(details.details.splits).map(([split, count]: [string, any], idx) => {
                    const total = Object.values(details.details.splits).reduce((a: any, b: any) => a + b, 0) as number;
                    const percent = total > 0 ? Math.round((count / total) * 100) : 0;
                    const colors = ['text-blue-400', 'text-purple-400', 'text-green-400'];
                    return (
                      <div key={split} className="flex-1 flex flex-col justify-center items-center p-4 bg-secondary/50 rounded-lg">
                        <span className={`text-2xl font-bold ${colors[idx % colors.length]}`}>{percent}%</span>
                        <span className="text-xs text-muted-foreground uppercase tracking-wider mt-1">{split}</span>
                        <span className="text-[10px] text-muted-foreground mt-1">{count.toLocaleString()} rows</span>
                      </div>
                    );
                  })
                ) : (
                  <p className="text-sm text-muted-foreground">No splits data available.</p>
                )}
              </div>
            </div>
          </div>

          {/* Data Preview */}
          <div className="bg-card border border-border rounded-xl flex flex-col flex-1 overflow-hidden min-h-[400px]">
            <div className="p-4 border-b border-border font-semibold flex items-center gap-2">
              <Table size={18}/> Data Preview {preview?.rows ? `(First ${preview.rows.length} rows)` : ''}
            </div>
            <div className="overflow-x-auto flex-1">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-muted-foreground uppercase bg-secondary/30 sticky top-0">
                  <tr>
                    <th className="px-6 py-3 border-b border-border w-16">#</th>
                    {preview?.columns?.map((col: string) => (
                      <th key={col} className="px-6 py-3 border-b border-border">{col}</th>
                    )) || <th className="px-6 py-3 border-b border-border">Data</th>}
                  </tr>
                </thead>
                <tbody>
                  {preview?.rows && preview.rows.length > 0 ? (
                    preview.rows.map((row: any[], idx: number) => (
                      <tr key={idx} className="border-b border-border hover:bg-secondary/20">
                        <td className="px-6 py-3 font-mono text-xs text-muted-foreground">{idx + 1}</td>
                        {row.map((cell: any, cellIdx: number) => (
                          <td key={cellIdx} className="px-6 py-3 max-w-md truncate">
                            {typeof cell === 'object' ? JSON.stringify(cell) : String(cell)}
                          </td>
                        ))}
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={10} className="px-6 py-8 text-center text-muted-foreground">
                        Preview data not available.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}