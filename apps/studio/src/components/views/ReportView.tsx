import { useState } from 'react';
import { useStore } from '../../store/useStore';
import { FileText, Download, Search } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const mockReportMarkdown = `
## Executive Summary
The recent fine-tuning of the Llama-3 model on \`training_split_v2.json\` yielded a **15% improvement** in overall F1 score across target categories.

### Key Metrics
| Metric | Baseline | Fine-Tuned | Delta |
|--------|----------|------------|-------|
| Accuracy | 0.72 | 0.84 | +0.12 |
| Precision | 0.68 | 0.81 | +0.13 |
| Recall | 0.70 | 0.86 | +0.16 |
| **F1 Score** | **0.69** | **0.83** | **+0.14** |
`;

export function ReportView() {
  const { reports, selectedEntityId, setView } = useStore();
  const [search, setSearch] = useState('');
  
  const report = reports.find(r => r.id === selectedEntityId);

  if (!selectedEntityId) {
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

  if (!report) return <div className="p-8 text-muted-foreground">Report not found</div>;

  return (
    <div className="h-full flex flex-col bg-background overflow-y-auto p-8">
      <div className="flex justify-between items-start mb-8 border-b border-border pb-6">
        <div>
          <h2 className="text-3xl font-bold flex items-center gap-3 mb-2">
            <FileText className="text-blue-500" /> {report.name}
          </h2>
          <p className="text-muted-foreground">Generated on {report.date}</p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-secondary hover:bg-secondary/80 rounded-md text-sm font-medium transition-colors">
          <Download size={16} /> Export PDF
        </button>
      </div>

      <div className="prose max-w-4xl prose-headings:text-foreground prose-a:text-blue-500 prose-table:border-collapse prose-th:border prose-th:border-border prose-th:p-2 prose-th:bg-secondary/50 prose-td:border prose-td:border-border prose-td:p-2 prose-tr:border-b prose-tr:border-border prose-pre:bg-secondary/50 prose-pre:text-foreground">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{mockReportMarkdown}</ReactMarkdown>
      </div>
    </div>
  );
}