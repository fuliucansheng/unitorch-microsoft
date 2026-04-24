import { useEffect, useRef } from 'react';
import { useStore, type EntityType } from '../../store/useStore';
import { ChatInput } from './ChatInput';
import { Bot, User, Loader2, FileIcon } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { cn } from '../../lib/utils';

export function ChatArea() {
  const { 
    sessions, activeSessionId, isTyping, 
    datasets, jobs, reports, labelTasks, setView 
  } = useStore();
  const endRef = useRef<HTMLDivElement>(null);

  const currentSession = sessions.find(s => s.id === activeSessionId);
  const messages = currentSession?.messages || [];

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const findEntity = (id: string) => {
    const dataset = datasets.find(d => d.id === id);
    if (dataset) return { type: 'dataset', name: dataset.name, id };
    
    const job = jobs.find(j => j.id === id);
    if (job) return { type: 'job', name: job.name, id };
    
    const report = reports.find(r => r.id === id);
    if (report) return { type: 'report', name: report.name, id };
    
    const labelTask = labelTasks.find(l => l.id === id);
    if (labelTask) return { type: 'label', name: labelTask.name, id };
    
    return null;
  };

  // Pre-process message to parse @entities into markdown links and highlight /commands
  const formatContent = (content: string) => {
    if (!content) return '';
    
    // Replace @entityId with a special markdown link pattern
    let formatted = content.replace(/@([a-zA-Z0-9_-]+)/g, (match, id) => {
      const entity = findEntity(id);
      if (entity) {
        return `[@${entity.name}](entity:${entity.type}:${entity.id})`;
      }
      return `\`${match}\``; // Fallback to inline code if entity not found
    });

    // Replace /commands with inline code
    formatted = formatted.replace(/(\/\w+(-\w+)*)/g, '`$1`');
    return formatted;
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return 'Uploaded';
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
  };

  // Extract system info injections to show them as attachments instead of raw text
  const processMessageData = (msg: any) => {
    let content = msg.content || '';
    let attachments = msg.attachments || [];

    const sysInfoStr = '[System Info: The user has uploaded the following files. Please use the paths below to access them:]';
    const sysInfoIndex = content.indexOf(sysInfoStr);
    
    if (sysInfoIndex !== -1) {
      const textBefore = content.substring(0, sysInfoIndex).trim();
      const textAfterSysInfo = content.substring(sysInfoIndex);
      
      // If attachments aren't already set (e.g. loaded from API history), parse them
      if (!attachments.length) {
        const lines = textAfterSysInfo.split('\n');
        const parsedAttachments = [];
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('- ') && trimmed.includes('(Server Path:')) {
            const name = trimmed.substring(2, trimmed.indexOf('(Server Path:')).trim();
            const cleanName = name.replace(/`/g, ''); // clean markdown artifacts
            parsedAttachments.push({
              id: Math.random().toString(),
              name: cleanName,
              size: 0,
              type: 'unknown'
            });
          }
        }
        attachments = parsedAttachments;
      }
      
      content = textBefore;
    }
    
    return { displayContent: content, displayAttachments: attachments };
  };

  return (
    <div className="flex flex-col h-full bg-background relative">
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg, index) => {
          const { displayContent, displayAttachments } = processMessageData(msg);

          return (
            <div 
              key={`${msg.id}-${index}`} 
              className={cn(
                "flex gap-4 max-w-3xl",
                msg.role === 'user' ? "ml-auto flex-row-reverse" : ""
              )}
            >
              <div className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1",
                msg.role === 'user' ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground"
              )}>
                {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
              </div>
              
              <div className={cn(
                "px-4 py-3 rounded-2xl text-sm flex flex-col gap-2",
                msg.role === 'user' ? "bg-primary text-primary-foreground" : "bg-secondary/50 text-foreground"
              )}>
                {displayContent && (
                  <div className={cn(
                    "prose max-w-none",
                    msg.role === 'user' 
                      ? "prose-p:text-primary-foreground prose-headings:text-primary-foreground prose-code:bg-black/20 prose-code:text-primary-foreground prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:before:content-none prose-code:after:content-none" 
                      : "prose-p:text-foreground prose-headings:text-foreground prose-code:bg-white prose-code:text-blue-600 prose-code:px-1.5 prose-code:py-0.5 prose-code:border prose-code:border-border prose-code:rounded-md prose-code:before:content-none prose-code:after:content-none"
                  )}>
                    <ReactMarkdown 
                      components={{
                        a: ({ node, href, children, ...props }) => {
                          if (href?.startsWith('entity:')) {
                            const [, type, id] = href.split(':');
                            return (
                              <button
                                onClick={(e) => {
                                  e.preventDefault();
                                  setView(type as EntityType, id);
                                }}
                                className={cn(
                                  "inline-flex items-center font-medium px-1.5 py-0.5 rounded transition-colors no-underline",
                                  msg.role === 'user' 
                                    ? "bg-white/20 hover:bg-white/30 text-white" 
                                    : "bg-blue-500/10 hover:bg-blue-500/20 text-blue-600"
                                )}
                              >
                                {children}
                              </button>
                            );
                          }
                          return <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>;
                        }
                      }}
                    >
                      {formatContent(displayContent)}
                    </ReactMarkdown>
                  </div>
                )}
                
                {/* Render Attachments */}
                {displayAttachments.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-1">
                    {displayAttachments.map((att: any) => (
                      <div 
                        key={att.id} 
                        className={cn(
                          "flex items-center gap-2 px-3 py-2 rounded-lg text-xs border",
                          msg.role === 'user' 
                            ? "bg-black/10 border-black/20 text-primary-foreground" 
                            : "bg-white border-border text-foreground"
                        )}
                      >
                        <FileIcon size={14} className={msg.role === 'user' ? "opacity-80" : "text-blue-500"} />
                        <span className="truncate max-w-[150px] font-medium">{att.name}</span>
                        <span className="opacity-60 text-[10px]">{formatFileSize(att.size)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
        
        {isTyping && (
          <div className="flex gap-4 max-w-3xl">
            <div className="w-8 h-8 rounded-full bg-secondary text-secondary-foreground flex items-center justify-center flex-shrink-0 mt-1">
              <Bot size={16} />
            </div>
            <div className="px-4 py-4 rounded-2xl bg-secondary/50 flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 size={16} className="animate-spin text-primary" />
              <span>Processing request...</span>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>
      
      <ChatInput />
    </div>
  );
}