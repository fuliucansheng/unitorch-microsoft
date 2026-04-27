import { useState, useEffect } from 'react';
import { useStore } from '../../store/useStore';
import { api } from '../../lib/api';
import { FileText, Download, Search, Loader2, Edit2, Trash2, Check, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const fallbackReportMarkdown = `
## Executive Summary
No detailed content available for this report.

### Key Metrics
| Metric | Baseline | Fine-Tuned | Delta |
|--------|----------|------------|-------|
| Accuracy | N/A | N/A | N/A |
`;

export function ReportView() {
  const { reports, selectedReportId, setView, initData } = useStore();
  const [search, setSearch] = useState('');
  
  const baseReport = reports.find(r => r.id === selectedReportId);
  const [reportDetails, setReportDetails] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (!selectedReportId) return;
    
    setIsEditing(false);
    setLoading(true);
    api.reports.getDetails(selectedReportId)
      .then(data => {
        setReportDetails(data);
        setEditContent(data.content || data.description || data.summary || fallbackReportMarkdown);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch report details:', err);
        setLoading(false);
      });
  }, [selectedReportId]);

  const handleDelete = async () => {
    if (!selectedReportId) return;
    if (!confirm('Are you sure you want to delete this report? This action cannot be undone.')) return;
    
    try {
      await api.reports.delete(selectedReportId);
      await initData();
      setView('report', null);
    } catch (err) {
      console.error('Failed to delete report:', err);
      alert('Failed to delete report');
    }
  };

  const handleSave = async () => {
    if (!selectedReportId) return;
    setIsSaving(true);
    try {
      const updated = await api.reports.update({
        id: selectedReportId,
        content: editContent
      });
      setReportDetails(updated);
      setIsEditing(false);
      await initData();
    } catch (err) {
      console.error('Failed to update report:', err);
      alert('Failed to update report');
    } finally {
      setIsSaving(false);
    }
  };

  if (!selectedReportId) {
    const filtered = reports.filter(r => r.name.toLowerCase().includes(search.toLowerCase()));
    return (
      <div className="h-full flex flex-col bg-background p-8">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold flex items-center gap-3">
            <FileText className="text-blue-500" /> Reports Overview
          </h2>
          <div className="relative w-64">
            <Search className="absolute left-3 top-2.5 text-muted-foreground" size={16} />
            <input 
              type="text" 
              placeholder="Search reports..." 
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-secondary/30 border border-border rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 overflow-y-auto">
          {filtered.map(r => (
            <div 
              key={r.id} 
              onClick={() => setView('report', r.id)}
              className="p-5 bg-card border border-border rounded-xl cursor-pointer hover:border-blue-500/50 transition-colors group"
            >
              <div className="flex items-start justify-between mb-2">
                <h3 className="font-semibold group-hover:text-blue-500 transition-colors truncate pr-2">{r.name}</h3>
                <FileText size={16} className="text-muted-foreground" />
              </div>
              <p className="text-xs text-muted-foreground mb-3 border-b border-border pb-3">{r.date}</p>
              <p className="text-sm text-foreground/80 line-clamp-2">{r.summary}</p>
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="col-span-full py-12 text-center text-muted-foreground">No reports found matching "{search}"</div>
          )}
        </div>
      </div>
    );
  }

  const report = reportDetails || baseReport;
  if (!report) return <div className="p-8 text-muted-foreground">Report not found</div>;

  const displayDate = report.created_at 
    ? new Date(report.created_at).toLocaleString() 
    : (report.date || 'Recent');

  return (
    <div className="h-full flex flex-col bg-background overflow-hidden p-8">
      <div className="flex justify-between items-start mb-6 border-b border-border pb-6 shrink-0">
        <div>
          <h2 className="text-3xl font-bold flex items-center gap-3 mb-2">
            <FileText className="text-blue-500" /> {report.name}
          </h2>
          <p className="text-muted-foreground">
            ID: <span className="font-mono">{report.id}</span> • Generated on {displayDate}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {isEditing ? (
            <>
              <button 
                title="Cancel"
                onClick={() => setIsEditing(false)}
                className="p-2 bg-secondary hover:bg-secondary/80 rounded-md transition-colors"
                disabled={isSaving}
              >
                <X size={18} />
              </button>
              <button 
                title="Save"
                onClick={handleSave}
                className="p-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-md transition-colors"
                disabled={isSaving}
              >
                {isSaving ? <Loader2 size={18} className="animate-spin" /> : <Check size={18} />}
              </button>
            </>
          ) : (
            <>
              <button 
                title="Edit"
                onClick={() => setIsEditing(true)}
                className="p-2 bg-secondary hover:bg-secondary/80 rounded-md transition-colors"
              >
                <Edit2 size={18} />
              </button>
              <button 
                title="Export PDF"
                className="p-2 bg-secondary hover:bg-secondary/80 rounded-md transition-colors"
              >
                <Download size={18} />
              </button>
              <button 
                title="Delete"
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
          <Loader2 className="animate-spin mr-2" size={24} /> Loading report content...
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto min-h-0">
          {isEditing ? (
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              className="w-full h-full min-h-[500px] p-4 bg-secondary/20 border border-border rounded-lg font-mono text-sm focus:outline-none focus:ring-1 focus:ring-primary resize-none"
            />
          ) : (
            <div className="prose max-w-4xl prose-headings:text-foreground prose-a:text-blue-500 prose-table:border-collapse prose-th:border prose-th:border-border prose-th:p-2 prose-th:bg-secondary/50 prose-td:border prose-td:border-border prose-td:p-2 prose-tr:border-b prose-tr:border-border prose-pre:bg-secondary/50 prose-pre:text-foreground pb-8">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {report.content || report.description || report.summary || fallbackReportMarkdown}
              </ReactMarkdown>
            </div>
          )}
        </div>
      )}
    </div>
  );
}