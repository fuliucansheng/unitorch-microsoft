import { useState, useEffect } from 'react';
import { useStore } from '../../store/useStore';
import { api } from '../../lib/api';
import { Database, Table, Search, Share2, Check, Loader2, List, Activity, Settings2, Edit2, X, Save, Trash2 } from 'lucide-react';

// Enhanced helper to robustly normalize API response
const normalizePreviewData = (data: any) => {
  if (!data) return { columns: [], rows: [], total: 0, raw: data };

  // Already formatted as { columns: [...], rows: [...] }
  if (Array.isArray(data.columns) && Array.isArray(data.rows)) {
    return { columns: data.columns, rows: data.rows, total: data.total ?? data.rows.length };
  }

  let items: any[] = [];
  if (Array.isArray(data)) {
    items = data;
  } else if (typeof data === 'object') {
    // Look for the first array in the object (e.g. data.items, data.rows, data.data)
    const arrays = Object.values(data).filter(Array.isArray);
    if (arrays.length > 0) {
      items = arrays[0] as any[];
    } else {
      // Treat the object itself as a single item if no array is found
      items = [data];
    }
  }

  if (items.length === 0) return { columns: [], rows: [], total: 0, raw: data };

  const extractDict = (item: any) => {
    if (item === null || typeof item !== 'object') return { value: item };
    if ('row' in item && typeof item.row === 'object') return item.row;
    if ('data' in item && typeof item.data === 'object') return item.data;
    return item;
  };

  const firstDict = extractDict(items[0]);
  const columns = Object.keys(firstDict);

  const rows = items.map(item => {
    const dict = extractDict(item);
    return columns.map(col => dict[col]);
  });

  return { columns, rows, total: data.total || items.length, raw: data };
};

export function DatasetView() {
  const { datasets, selectedDatasetId, setView, initData } = useStore();
  const [search, setSearch] = useState('');
  const [copied, setCopied] = useState(false);
  
  const baseDataset = datasets.find(d => d.id === selectedDatasetId);
  const [details, setDetails] = useState<any>(null);
  const [preview, setPreview] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Edit State
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [editForm, setEditForm] = useState({
    name: '',
    description: '',
    task_type: '',
    metrics: '',
    label_columns: ''
  });

  const loadDatasetDetails = () => {
    if (!selectedDatasetId) return;
    setLoading(true);
    setPreview(null);
    
    // 1. Fetch details
    api.datasets.getDetails(selectedDatasetId)
      .then(setDetails)
      .catch(console.error);

    // 2. Fetch preview (no strict split parameter, just get default samples)
    api.datasets.getPreview(selectedDatasetId)
      .then(data => {
        setPreview(normalizePreviewData(data));
      })
      .catch(err => {
        console.error('Preview fetch error:', err);
        setPreview({ error: err.message });
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    setIsEditing(false);
    loadDatasetDetails();
  }, [selectedDatasetId]);

  const handleShare = () => {
    navigator.clipboard.writeText(`${window.location.origin}/datasets/${selectedDatasetId}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const startEditing = () => {
    const dataset = details || baseDataset;
    const meta = dataset?.meta || {};
    setEditForm({
      name: dataset?.name || '',
      description: dataset?.description || '',
      task_type: meta.task_type || '',
      metrics: meta.metrics ? meta.metrics.join(', ') : '',
      label_columns: meta.label_columns ? meta.label_columns.join(', ') : ''
    });
    setIsEditing(true);
  };

  const saveMetadata = async () => {
    if (!selectedDatasetId) return;
    setIsSaving(true);
    try {
      const metaPayload = {
        name: editForm.name.trim(),
        description: editForm.description.trim(),
        task_type: editForm.task_type.trim(),
        metrics: editForm.metrics.split(',').map(s => s.trim()).filter(Boolean),
        label_columns: editForm.label_columns.split(',').map(s => s.trim()).filter(Boolean)
      };

      const updated = await api.datasets.updateMeta(selectedDatasetId, metaPayload);
      setDetails(updated);
      setIsEditing(false);
      await initData(); // Refresh sidebar datasets list
    } catch (err) {
      console.error('Failed to update metadata', err);
      alert('Failed to update metadata');
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedDatasetId) return;
    if (!confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) return;
    
    try {
      await api.datasets.delete(selectedDatasetId);
      await initData();
      setView('dataset', null);
    } catch (err) {
      console.error('Failed to delete dataset:', err);
      alert('Failed to delete dataset');
    }
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

  const meta = dataset.meta || {};
  
  // Auto-infer columns if they are not explicitly defined in metadata
  const definedColumns = meta.columns || dataset.columns || dataset.schema || [];
  const displayColumns = definedColumns.length > 0 
    ? definedColumns 
    : (preview?.columns?.map((c: string) => ({ name: c, type: 'auto', description: 'Inferred from data' })) || []);

  return (
    <div className="h-full flex flex-col bg-background overflow-y-auto p-8 space-y-8">
      <div className="flex justify-between items-start gap-4">
        <div className="flex-1 min-w-0">
          {isEditing ? (
            <div className="space-y-3">
              <input 
                value={editForm.name}
                onChange={e => setEditForm(prev => ({...prev, name: e.target.value}))}
                placeholder="Dataset Name"
                className="w-full text-2xl font-bold bg-secondary/30 border border-border rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary"
              />
              <p className="text-muted-foreground text-sm">
                ID: {dataset.id} • {dataset.rows ? dataset.rows.toLocaleString() : 0} rows • {dataset.size || 'Unknown'}
              </p>
              <textarea 
                value={editForm.description}
                onChange={e => setEditForm(prev => ({...prev, description: e.target.value}))}
                placeholder="Dataset Description"
                rows={2}
                className="w-full text-sm bg-secondary/30 border border-border rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-primary resize-none mt-2"
              />
            </div>
          ) : (
            <div>
              <h2 className="text-2xl font-bold flex items-center gap-3 mb-2 truncate">
                <Database className="text-blue-400 shrink-0" /> {dataset.name}
              </h2>
              <p className="text-muted-foreground text-sm">
                ID: {dataset.id} • {dataset.rows ? dataset.rows.toLocaleString() : 0} rows • {dataset.size || 'Unknown'}
              </p>
              {dataset.description && (
                <p className="text-sm mt-2 text-foreground/80">{dataset.description}</p>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {isEditing ? (
            <>
              <button 
                title="Cancel"
                onClick={() => setIsEditing(false)}
                disabled={isSaving}
                className="p-2 bg-secondary hover:bg-secondary/80 rounded-md transition-colors"
              >
                <X size={18} />
              </button>
              <button 
                title="Save"
                onClick={saveMetadata}
                disabled={isSaving}
                className="p-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-md transition-colors disabled:opacity-50"
              >
                {isSaving ? <Loader2 size={18} className="animate-spin" /> : <Save size={18} />}
              </button>
            </>
          ) : (
            <>
              <button 
                title="Edit"
                onClick={startEditing}
                className="p-2 bg-secondary hover:bg-secondary/80 rounded-md transition-colors"
              >
                <Edit2 size={18} />
              </button>
              <button 
                title="Share"
                onClick={handleShare}
                className="p-2 bg-card border border-border hover:bg-secondary rounded-md transition-colors"
              >
                {copied ? <Check size={18} className="text-green-500" /> : <Share2 size={18} className="text-blue-400" />}
              </button>
              <button 
                title="Delete Dataset"
                onClick={handleDelete}
                className="p-2 bg-destructive/10 text-destructive hover:bg-destructive/20 rounded-md transition-colors"
              >
                <Trash2 size={18} />
              </button>
            </>
          )}
        </div>
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center text-muted-foreground">
          <Loader2 className="animate-spin mr-2" size={24} /> Loading dataset details...
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Meta Info */}
            <div className="bg-card border border-border rounded-xl p-6">
              <h3 className="font-semibold mb-4 flex items-center gap-2"><Settings2 size={18}/> Metadata</h3>
              
              {isEditing ? (
                <div className="space-y-4">
                  <div>
                    <label className="text-xs text-muted-foreground font-medium mb-1 block">Task Type</label>
                    <input 
                      value={editForm.task_type}
                      onChange={e => setEditForm(prev => ({...prev, task_type: e.target.value}))}
                      className="w-full text-sm bg-secondary/30 border border-border rounded px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary"
                      placeholder="e.g. classification"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-muted-foreground font-medium mb-1 block">Metrics (comma separated)</label>
                    <input 
                      value={editForm.metrics}
                      onChange={e => setEditForm(prev => ({...prev, metrics: e.target.value}))}
                      className="w-full text-sm bg-secondary/30 border border-border rounded px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary"
                      placeholder="e.g. accuracy, f1"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-muted-foreground font-medium mb-1 block">Label Columns (comma separated)</label>
                    <input 
                      value={editForm.label_columns}
                      onChange={e => setEditForm(prev => ({...prev, label_columns: e.target.value}))}
                      className="w-full text-sm bg-secondary/30 border border-border rounded px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary"
                      placeholder="e.g. target, label"
                    />
                  </div>
                </div>
              ) : (
                <div className="space-y-4 text-sm">
                  <div>
                    <span className="text-muted-foreground block mb-1">Task Type</span>
                    <span className="px-2 py-1 bg-secondary rounded-md capitalize">{meta.task_type || 'Unspecified'}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block mb-1">Created At</span>
                    <span>{dataset.created_at ? new Date(dataset.created_at).toLocaleString() : 'N/A'}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Columns & Labels */}
            <div className="bg-card border border-border rounded-xl p-6 md:col-span-2">
              <h3 className="font-semibold mb-4 flex items-center gap-2"><List size={18}/> Columns Definition</h3>
              <div className="flex flex-col gap-4">
                {displayColumns && displayColumns.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="text-xs text-muted-foreground bg-secondary">
                        <tr>
                          <th className="px-4 py-2 border-b border-border">Name</th>
                          <th className="px-4 py-2 border-b border-border">Type</th>
                          <th className="px-4 py-2 border-b border-border">Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        {displayColumns.map((col: any, idx: number) => {
                          const isLabel = isEditing 
                            ? editForm.label_columns.split(',').map(s => s.trim()).includes(col.name)
                            : meta.label_columns?.includes(col.name);
                            
                          return (
                            <tr key={idx} className="border-b border-border">
                              <td className="px-4 py-2 font-medium">
                                {col.name}
                                {isLabel && (
                                  <span className="ml-2 px-1.5 py-0.5 bg-blue-500/10 text-blue-500 text-[10px] rounded uppercase">Label</span>
                                )}
                              </td>
                              <td className="px-4 py-2 text-muted-foreground">{col.type}</td>
                              <td className="px-4 py-2 text-muted-foreground">{col.description || '-'}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No columns available or inferred.</p>
                )}
                
                {((isEditing && editForm.metrics) || (!isEditing && meta.metrics && meta.metrics.length > 0)) && (
                  <div>
                    <span className="text-xs text-muted-foreground uppercase tracking-wider block mb-2"><Activity size={14} className="inline mr-1"/>Metrics</span>
                    <div className="flex flex-wrap gap-2">
                      {(isEditing ? editForm.metrics.split(',').map(s=>s.trim()).filter(Boolean) : meta.metrics).map((m: string) => (
                        <span key={m} className="px-2 py-1 bg-secondary text-xs rounded-md">{m}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Data Preview */}
          <div className="bg-card border border-border rounded-xl flex flex-col flex-1 overflow-hidden min-h-[400px]">
            <div className="p-4 border-b border-border font-semibold flex items-center gap-2">
              <Table size={18}/> Data Preview {preview?.rows && preview.rows.length > 0 ? `(First ${preview.rows.length} rows)` : ''} 
              {preview?.total !== undefined && preview?.total > 0 && <span className="text-muted-foreground font-normal ml-2">of {preview.total} total</span>}
            </div>
            <div className="overflow-x-auto flex-1">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-muted-foreground uppercase bg-secondary sticky top-0 z-10">
                  <tr>
                    <th className="px-6 py-3 border-b border-border w-16">#</th>
                    {preview?.columns?.map((col: string) => (
                      <th key={col} className="px-6 py-3 border-b border-border">{col}</th>
                    )) || <th className="px-6 py-3 border-b border-border">Data</th>}
                  </tr>
                </thead>
                <tbody>
                  {preview?.error ? (
                    <tr>
                      <td colSpan={10} className="px-6 py-8 text-center text-destructive">
                        Failed to load preview: {preview.error}
                      </td>
                    </tr>
                  ) : preview?.rows && preview.rows.length > 0 ? (
                    preview.rows.map((row: any[], idx: number) => (
                      <tr key={idx} className="border-b border-border hover:bg-secondary/20">
                        <td className="px-6 py-3 font-mono text-xs text-muted-foreground">{idx + 1}</td>
                        {row.map((cell: any, cellIdx: number) => (
                          <td key={cellIdx} className="px-6 py-3 max-w-md truncate" title={typeof cell === 'object' ? JSON.stringify(cell) : String(cell)}>
                            {typeof cell === 'object' ? JSON.stringify(cell) : String(cell)}
                          </td>
                        ))}
                      </tr>
                    ))
                  ) : preview?.raw ? (
                    <tr>
                      <td colSpan={10} className="p-0">
                        <pre className="p-6 text-xs overflow-x-auto max-h-[400px]">
                          {JSON.stringify(preview.raw, null, 2)}
                        </pre>
                      </td>
                    </tr>
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