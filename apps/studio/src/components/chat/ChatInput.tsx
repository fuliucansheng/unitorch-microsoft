import { useState, useRef, type KeyboardEvent, type ChangeEvent } from 'react';
import { useStore } from '../../store/useStore';
import { Send, Command, Database, Paperclip, X, FileIcon, TerminalSquare, Tag, FileText } from 'lucide-react';

const COMMANDS = ['/create-dataset', '/process-dataset', '/create-job', '/improve-prompt'];

export function ChatInput() {
  const [input, setInput] = useState('');
  const [showCommands, setShowCommands] = useState(false);
  const [showEntities, setShowEntities] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [files, setFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { addMessage, datasets, jobs, labelTasks, reports, setTyping } = useStore();

  const handleSend = () => {
    if (!input.trim() && files.length === 0) return;

    const userMessage = input.trim();
    const attachments = files.map(f => ({
      id: Math.random().toString(36).substring(2, 9),
      name: f.name,
      size: f.size,
      type: f.type
    }));

    addMessage({ role: 'user', content: userMessage, attachments });
    
    if (userMessage) {
      setHistory(prev => [...prev, userMessage]);
    }
    setHistoryIndex(-1);
    
    setInput('');
    setFiles([]);
    setShowCommands(false);
    setShowEntities(false);
    setTyping(true);
    
    // Simulate AI response processing
    setTimeout(() => {
      addMessage({ 
        role: 'assistant', 
        content: userMessage 
          ? `Executed \`${userMessage}\`\n\nI have successfully queued this workflow.` 
          : `I received your file${attachments.length > 1 ? 's' : ''}. Processing now...`
      });
      setTyping(false);
    }, 1500);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
      return;
    }
    
    if (input === '') {
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (history.length > 0) {
          const nextIndex = historyIndex < history.length - 1 ? historyIndex + 1 : historyIndex;
          setHistoryIndex(nextIndex);
          setInput(history[history.length - 1 - nextIndex]);
        }
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (historyIndex > 0) {
          const nextIndex = historyIndex - 1;
          setHistoryIndex(nextIndex);
          setInput(history[history.length - 1 - nextIndex]);
        } else if (historyIndex === 0) {
          setHistoryIndex(-1);
          setInput('');
        }
      }
    }
  };

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setInput(val);
    
    const lastWord = val.split(/[\s\n]+/).pop() || '';
    setShowCommands(lastWord.startsWith('/'));
    setShowEntities(lastWord.startsWith('@'));
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFiles(prev => [...prev, ...Array.from(e.target.files!)]);
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const insertText = (text: string) => {
    const words = input.split(/[\s\n]+/);
    words.pop();
    setInput([...words, text, ''].join(' ') + ' ');
    setShowCommands(false);
    setShowEntities(false);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
  };

  const allEntities = [
    ...datasets.map(d => ({ ...d, type: 'dataset', icon: Database, color: 'text-blue-500' })),
    ...jobs.map(j => ({ ...j, type: 'job', icon: TerminalSquare, color: 'text-orange-500' })),
    ...labelTasks.map(l => ({ ...l, type: 'label', icon: Tag, color: 'text-purple-500' })),
    ...reports.map(r => ({ ...r, type: 'report', icon: FileText, color: 'text-green-500' }))
  ];

  // 计算当前的搜索词并过滤实体
  const lastWord = input.split(/[\s\n]+/).pop() || '';
  const mentionQuery = lastWord.startsWith('@') ? lastWord.slice(1).toLowerCase() : '';
  const filteredEntities = allEntities.filter(e => 
    e.name.toLowerCase().includes(mentionQuery) || 
    e.id.toLowerCase().includes(mentionQuery)
  );

  return (
    <div className="p-4 bg-background/80 backdrop-blur-md border-t border-border flex flex-col items-center">
      <div className="w-full max-w-3xl relative flex flex-col gap-2">
        {/* Autocomplete menus */}
        {showCommands && (
          <div className="absolute bottom-full left-0 mb-2 w-64 bg-card border border-border rounded-xl shadow-xl overflow-hidden z-10 animate-in fade-in slide-in-from-bottom-2">
            <div className="px-3 py-2 text-xs font-medium text-muted-foreground bg-secondary/50 flex justify-between">
              <span>Commands</span>
              <span className="text-[10px]">Enter to select</span>
            </div>
            <div className="p-1 max-h-48 overflow-y-auto">
              {COMMANDS.map(cmd => (
                <button
                  key={cmd}
                  onClick={() => insertText(cmd)}
                  className="w-full text-left px-3 py-2 text-sm hover:bg-secondary rounded-lg flex items-center gap-2 transition-colors"
                >
                  <Command size={14} className="text-primary" /> {cmd}
                </button>
              ))}
            </div>
          </div>
        )}

        {showEntities && (
          <div className="absolute bottom-full left-0 mb-2 w-64 bg-card border border-border rounded-xl shadow-xl overflow-hidden z-10 animate-in fade-in slide-in-from-bottom-2">
            <div className="px-3 py-2 text-xs font-medium text-muted-foreground bg-secondary/50 flex justify-between">
              <span>Mentions</span>
              {mentionQuery && <span className="text-[10px]">Searching "{mentionQuery}"</span>}
            </div>
            <div className="p-1 max-h-48 overflow-y-auto">
              {filteredEntities.length > 0 ? (
                filteredEntities.map(entity => (
                  <button
                    key={`${entity.type}-${entity.id}`}
                    onClick={() => insertText(`@${entity.id}`)}
                    className="w-full text-left px-3 py-2 text-sm hover:bg-secondary rounded-lg flex items-center gap-2 transition-colors"
                  >
                    <entity.icon size={14} className={entity.color} shrink-0 /> 
                    <span className="truncate">{entity.name}</span>
                  </button>
                ))
              ) : (
                <div className="px-3 py-4 text-center text-sm text-muted-foreground">
                  No matches found
                </div>
              )}
            </div>
          </div>
        )}

        {/* File Previews */}
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 px-1 mb-1">
            {files.map((file, idx) => (
               <div key={idx} className="flex items-center gap-2 bg-card border border-border shadow-sm rounded-lg px-3 py-2 text-xs group relative pr-8 animate-in zoom-in-95">
                <div className="p-1.5 bg-blue-500/10 rounded-md text-blue-500">
                  <FileIcon size={14} />
                </div>
                <div className="flex flex-col">
                  <span className="truncate max-w-[150px] font-medium">{file.name}</span>
                  <span className="text-muted-foreground text-[10px]">{formatFileSize(file.size)}</span>
                </div>
                <button 
                  onClick={() => removeFile(idx)}
                  className="absolute right-2 p-1 rounded-full hover:bg-destructive hover:text-destructive-foreground text-muted-foreground transition-colors"
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="relative flex items-end gap-2 bg-card border border-border rounded-2xl p-2 shadow-sm focus-within:ring-1 focus-within:ring-primary focus-within:border-primary transition-all">
          <input 
            type="file" 
            multiple 
            ref={fileInputRef} 
            className="hidden" 
            onChange={handleFileChange} 
          />
          
          <button 
            onClick={() => fileInputRef.current?.click()}
            className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded-xl transition-colors shrink-0 mb-0.5"
            title="Attach files"
          >
            <Paperclip size={20} />
          </button>
          
          <textarea
            rows={1}
            value={input}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask Ads Studio anything... (Shift+Enter for new line)"
            className="w-full bg-transparent border-none py-2.5 px-1 text-sm focus:outline-none resize-none max-h-[200px] min-h-[44px]"
            style={{ fieldSizing: 'content' } as React.CSSProperties} // Allows auto-grow in supported browsers
          />
          
          <button 
            onClick={handleSend}
            disabled={!input.trim() && files.length === 0}
            title="Send message"
            className={`p-2 rounded-xl transition-all shrink-0 mb-0.5 flex items-center justify-center ${
              (input.trim() || files.length > 0) 
                ? 'bg-primary text-primary-foreground shadow-md hover:bg-primary/90 hover:scale-105' 
                : 'bg-secondary text-muted-foreground opacity-50'
            }`}
          >
            <Send size={18} className={(input.trim() || files.length > 0) ? 'ml-0.5' : ''} />
          </button>
        </div>
        <div className="text-center mt-1 text-[10px] text-muted-foreground">
          Ads Studio can make mistakes. Check important info.
        </div>
      </div>
    </div>
  );
}