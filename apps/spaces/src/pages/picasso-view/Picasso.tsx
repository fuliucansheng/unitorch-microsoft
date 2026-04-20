import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { PanelLeftClose, PanelLeftOpen, Microscope } from 'lucide-react';

const Picasso = () => {
  // 在非大屏 (小于 1024px) 上默认处于收缩状态
  const [isCollapsed, setIsCollapsed] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.innerWidth < 1024;
    }
    return false;
  });

  return (
    <div className="flex flex-col md:flex-row flex-1 h-full overflow-hidden bg-transparent">
      {/* 侧边栏/移动端顶部导航 */}
      <aside 
        className={`flex-shrink-0 border-b md:border-b-0 md:border-r border-black/[0.04] dark:border-white/[0.05] bg-white/40 dark:bg-zinc-950/30 backdrop-blur-2xl flex flex-row md:flex-col p-3 md:p-4 gap-2 overflow-x-auto no-scrollbar items-center md:items-stretch transition-all duration-300 ease-in-out w-full ${
          isCollapsed ? 'md:w-[76px]' : 'md:w-64'
        }`}
      >
        {/* 顶部标题与折叠按钮 (仅桌面端显示) */}
        <div className="hidden md:flex items-center justify-between mb-4 px-2">
          <div 
            className={`text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider transition-all duration-300 overflow-hidden whitespace-nowrap ${
              isCollapsed ? 'max-w-0 opacity-0' : 'max-w-[150px] opacity-100'
            }`}
          >
            Picasso Menu
          </div>
          <button 
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-black/5 dark:hover:bg-white/10 transition-colors shrink-0"
            title={isCollapsed ? "Expand Menu" : "Collapse Menu"}
          >
            {isCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </button>
        </div>
        
        {/* 导航菜单项 */}
        <NavLink 
          to="/picasso/overview" 
          title="Overview"
          className={({ isActive }) => 
            `flex items-center whitespace-nowrap gap-2 px-3 py-2 md:py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
              isActive 
                ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]' 
                : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
            } ${isCollapsed ? 'md:justify-center md:px-0' : ''}`
          }
        >
          <span className="text-base md:text-lg flex-shrink-0">🌐</span>
          <span 
            className={`transition-all duration-300 overflow-hidden ${
              isCollapsed ? 'md:max-w-0 md:opacity-0 md:ml-0' : 'md:max-w-[150px] md:opacity-100 md:ml-1'
            }`}
          >
            Overview
          </span>
        </NavLink>

        {/* Examples 菜单组 */}
        <div className="flex flex-row md:flex-col gap-2 md:gap-1">
          <NavLink 
            to="/picasso/examples" 
            title="Examples"
            className={({ isActive }) => 
              `flex items-center whitespace-nowrap gap-2 px-3 py-2 md:py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
                isActive 
                  ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]' 
                  : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
              } ${isCollapsed ? 'md:justify-center md:px-0' : ''}`
            }
          >
            <span className="text-base md:text-lg flex-shrink-0">🎢</span>
            <span 
              className={`transition-all duration-300 overflow-hidden ${
                isCollapsed ? 'md:max-w-0 md:opacity-0 md:ml-0' : 'md:max-w-[150px] md:opacity-100 md:ml-1'
              }`}
            >
              Examples
            </span>
          </NavLink>

          {/* 二级菜单: DR Measurement */}
          <NavLink 
            to="/picasso/dr-measurement" 
            title="DR Measurement"
            className={({ isActive }) => 
              `flex items-center whitespace-nowrap gap-2 px-3 py-2 md:py-2 rounded-xl text-[13px] font-medium transition-all duration-300 ${
                isActive 
                  ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]' 
                  : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
              } ${isCollapsed ? 'md:justify-center md:px-0' : 'md:ml-6'}`
            }
          >
            <span className="text-base md:text-lg flex-shrink-0 flex items-center justify-center w-[20px]"><Microscope size={16} /></span>
            <span 
              className={`transition-all duration-300 overflow-hidden ${
                isCollapsed ? 'md:max-w-0 md:opacity-0 md:ml-0' : 'md:max-w-[150px] md:opacity-100 md:ml-1'
              }`}
            >
              DR Measurement
            </span>
          </NavLink>
        </div>
      </aside>

      {/* 内容区 */}
      <main className="flex-1 overflow-y-auto p-4 md:p-8 relative custom-scrollbar">
        {/* 更优雅的渐变氛围背景 */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-100/40 via-transparent to-transparent dark:from-indigo-900/10 pointer-events-none" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-purple-100/40 via-transparent to-transparent dark:from-purple-900/10 pointer-events-none" />
        <div className="relative z-10 h-full">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default Picasso;