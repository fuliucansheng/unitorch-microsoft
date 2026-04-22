import { useState, useEffect } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { PanelLeftClose, PanelLeftOpen, Microscope, LayoutDashboard, Layers, Focus, Menu, X } from 'lucide-react';

const Picasso = () => {
  // 桌面端侧边栏折叠状态
  const [isDesktopCollapsed, setIsDesktopCollapsed] = useState(false);
  // 移动端侧边栏（抽屉）展开状态
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const location = useLocation();

  // 路由变化时自动收起移动端侧边栏
  useEffect(() => {
    setIsMobileOpen(false);
  }, [location.pathname]);

  return (
    <div className="flex flex-1 h-full relative overflow-hidden bg-transparent">
      
      {/* ================= 移动端半透明遮罩 ================= */}
      {isMobileOpen && (
        <div 
          className="md:hidden absolute inset-0 bg-zinc-900/20 dark:bg-black/40 backdrop-blur-sm z-30 transition-opacity"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* ================= 移动端唤出按钮 ================= */}
      {!isMobileOpen && (
        <button
          onClick={() => setIsMobileOpen(true)}
          className="md:hidden absolute top-4 left-0 z-20 py-2 pl-2 pr-3 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md border border-l-0 border-black/[0.08] dark:border-white/[0.08] rounded-r-xl shadow-sm text-zinc-600 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
        >
          <Menu size={20} />
        </button>
      )}

      {/* ================= 统一的左侧边栏 (桌面端 Static / 移动端 Drawer) ================= */}
      <aside 
        className={`
          absolute md:relative z-40 h-full flex flex-col flex-shrink-0 
          border-r border-black/[0.04] dark:border-white/[0.05] 
          bg-white/95 md:bg-white/40 dark:bg-zinc-950/95 md:dark:bg-zinc-950/30 md:backdrop-blur-2xl backdrop-blur-3xl
          transition-all duration-300 ease-in-out p-4 shadow-2xl md:shadow-none
          ${isMobileOpen ? 'translate-x-0 w-64' : '-translate-x-full w-64 md:translate-x-0'}
          ${isDesktopCollapsed ? 'md:w-[76px] md:items-center' : 'md:w-64 md:items-stretch'}
        `}
      >
        {/* 顶部标题与控制按钮 */}
        <div className={`flex items-center mb-4 transition-all duration-300 ${isDesktopCollapsed ? 'md:justify-center px-0' : 'justify-between px-2'}`}>
          <div 
            className={`text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider transition-all duration-300 overflow-hidden whitespace-nowrap ${
              isDesktopCollapsed ? 'md:max-w-0 md:opacity-0' : 'max-w-[150px] opacity-100'
            }`}
          >
            Picasso Menu
          </div>
          
          {/* 桌面端折叠按钮 */}
          <button 
            onClick={() => setIsDesktopCollapsed(!isDesktopCollapsed)}
            className="hidden md:flex items-center justify-center w-8 h-8 rounded-lg text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-black/5 dark:hover:bg-white/10 transition-colors shrink-0"
            title={isDesktopCollapsed ? "Expand Menu" : "Collapse Menu"}
          >
            {isDesktopCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </button>

          {/* 移动端关闭按钮 */}
          <button 
            onClick={() => setIsMobileOpen(false)}
            className="md:hidden flex items-center justify-center w-8 h-8 rounded-lg text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-black/5 dark:hover:bg-white/10 transition-colors shrink-0"
          >
            <X size={20} />
          </button>
        </div>
        
        {/* 导航菜单项 */}
        <NavLink 
          to="/picasso/overview" 
          title="Overview"
          className={({ isActive }) => 
            `flex items-center whitespace-nowrap px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
              isActive 
                ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]' 
                : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
            } ${isDesktopCollapsed ? 'md:justify-center md:px-0 md:w-10 md:h-10' : ''}`
          }
        >
          <span className="flex-shrink-0 flex items-center justify-center w-5 h-5 text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-zinc-100">
            <LayoutDashboard size={18} />
          </span>
          <span 
            className={`transition-all duration-300 overflow-hidden ${
              isDesktopCollapsed ? 'md:max-w-0 md:opacity-0 md:ml-0' : 'max-w-[150px] opacity-100 ml-2'
            }`}
          >
            Overview
          </span>
        </NavLink>

        {/* Spaces 菜单组 */}
        <div className={`flex flex-col gap-1 mt-1 group/spaces ${isDesktopCollapsed ? 'md:items-center' : ''}`}>
          <NavLink 
            to="/picasso/examples" 
            title="Spaces"
            className={({ isActive }) => 
              `flex items-center whitespace-nowrap px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
                isActive 
                  ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]' 
                  : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
              } ${isDesktopCollapsed ? 'md:justify-center md:px-0 md:w-10 md:h-10' : ''}`
            }
          >
            <span className="flex-shrink-0 flex items-center justify-center w-5 h-5 text-zinc-500 dark:text-zinc-400 group-hover/spaces:text-zinc-900 dark:group-hover/spaces:text-zinc-100">
              <Layers size={18} />
            </span>
            <span 
              className={`transition-all duration-300 overflow-hidden ${
                isDesktopCollapsed ? 'md:max-w-0 md:opacity-0 md:ml-0' : 'max-w-[150px] opacity-100 ml-2'
              }`}
            >
              Spaces
            </span>
          </NavLink>

          {/* 二级菜单: DR Measurement */}
          <NavLink 
            to="/picasso/dr-measurement" 
            title="DR Measurement"
            className={({ isActive }) => 
              `flex items-center whitespace-nowrap px-3 py-2 rounded-xl text-[13px] font-medium transition-all duration-300 ${
                isActive 
                  ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]' 
                  : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
              } ${isDesktopCollapsed ? 'md:justify-center md:px-0 md:w-10 md:h-10 md:ml-0' : 'ml-6'}`
            }
          >
            <span className="flex-shrink-0 flex items-center justify-center w-5 h-5 text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-zinc-100">
              <Microscope size={18} />
            </span>
            <span 
              className={`transition-all duration-300 overflow-hidden ${
                isDesktopCollapsed ? 'md:max-w-0 md:opacity-0 md:ml-0' : 'max-w-[150px] opacity-100 ml-2'
              }`}
            >
              DR Measurement
            </span>
          </NavLink>

          {/* 二级菜单: ROI Detection */}
          <NavLink 
            to="/picasso/roi-detection" 
            title="ROI Detection"
            className={({ isActive }) => 
              `flex items-center whitespace-nowrap px-3 py-2 rounded-xl text-[13px] font-medium transition-all duration-300 ${
                isActive 
                  ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]' 
                  : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
              } ${isDesktopCollapsed ? 'md:justify-center md:px-0 md:w-10 md:h-10 md:ml-0' : 'ml-6'}`
            }
          >
            <span className="flex-shrink-0 flex items-center justify-center w-5 h-5 text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-zinc-100">
              <Focus size={18} />
            </span>
            <span 
              className={`transition-all duration-300 overflow-hidden ${
                isDesktopCollapsed ? 'md:max-w-0 md:opacity-0 md:ml-0' : 'max-w-[150px] opacity-100 ml-2'
              }`}
            >
              ROI Detection
            </span>
          </NavLink>
        </div>
      </aside>

      {/* ================= 内容区 ================= */}
      <main className="flex-1 overflow-y-auto p-4 pt-16 md:p-8 relative custom-scrollbar">
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