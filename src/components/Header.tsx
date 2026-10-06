import React, { useState } from 'react';
import { Scan, Menu, X, ArrowRight, Activity } from 'lucide-react';

export type NavTab = 'home' | 'analyze' | 'how-it-works' | 'research' | 'history';

interface HeaderProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  backendConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  backendConnected,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems: { id: NavTab; label: string }[] = [
    { id: 'home', label: 'Home' },
    { id: 'analyze', label: 'Workstation' },
    { id: 'how-it-works', label: 'How It Works' },
    { id: 'research', label: 'Research' },
    { id: 'history', label: 'History' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-950/90 backdrop-blur-xl border-b border-slate-800/80 w-full">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 h-14 sm:h-16 flex items-center justify-between gap-2">
        {/* Left: Product Branding */}
        <div
          onClick={() => onTabChange('home')}
          className="flex items-center gap-2.5 sm:gap-3 cursor-pointer group select-none min-w-0"
        >
          <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center text-cyan-400 group-hover:border-cyan-400 transition-colors shadow-sm shrink-0">
            <Scan className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5 sm:gap-2">
              <span className="font-extrabold text-xs sm:text-base tracking-tight text-white flex items-center gap-1 truncate">
                <span className="text-cyan-400">◉</span> EMOTION DETECTOR
              </span>
              <span className="hidden sm:inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-mono font-medium bg-slate-900 text-slate-400 border border-slate-800">
                A4 ONNX
              </span>
            </div>
            <p className="text-[9px] sm:text-[10px] font-mono text-slate-400 tracking-wider hidden sm:block uppercase truncate">
              REAL-TIME FACIAL EXPRESSION ANALYSIS
            </p>
          </div>
        </div>

        {/* Center Navigation: Desktop / Large Tablet */}
        <nav className="hidden md:flex items-center gap-1 p-1 rounded-lg bg-slate-900/60 border border-slate-800/80">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`min-h-[36px] px-3.5 py-1.5 rounded-md text-xs font-medium transition-all focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                  isActive
                    ? 'bg-cyan-400 text-slate-950 font-bold shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Right Actions */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Backend Status indicator representing REAL health check */}
          <div className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800 text-[10px] sm:text-[11px] font-mono">
            <span
              className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                backendConnected
                  ? 'bg-emerald-400 shadow-[0_0_6px_#34d399]'
                  : 'bg-rose-500 shadow-[0_0_6px_rgba(244,63,94,0.6)]'
              }`}
            />
            <span className={backendConnected ? 'text-emerald-300 font-semibold truncate' : 'text-slate-400 truncate'}>
              {backendConnected ? 'BACKEND ●' : 'OFFLINE'}
            </span>
          </div>

          {/* Launch Analyzer CTA on Desktop/Tablet */}
          <button
            onClick={() => onTabChange('analyze')}
            className="hidden sm:flex min-h-[36px] px-3.5 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold tracking-wide transition-all shadow-[0_0_12px_rgba(6,182,212,0.3)] items-center gap-1.5 active:scale-95 focus-visible:outline-2 focus-visible:outline-cyan-400"
          >
            <span>WORKSTATION</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>

          {/* Mobile Hamburger Toggle (minimum 44px tap target) */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden min-h-[44px] min-w-[44px] p-2.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-900 border border-slate-800 flex items-center justify-center active:scale-95 focus-visible:outline-2 focus-visible:outline-cyan-400"
            aria-label="Toggle navigation menu"
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer with >= 44px touch targets */}
      {mobileMenuOpen && (
        <div className="md:hidden border-b border-slate-800 bg-slate-950/98 px-4 pt-2 pb-4 space-y-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                onTabChange(item.id);
                setMobileMenuOpen(false);
              }}
              className={`w-full min-h-[44px] text-left px-4 py-2.5 rounded-lg text-xs font-medium transition-colors flex items-center justify-between ${
                activeTab === item.id
                  ? 'bg-cyan-400 text-slate-950 font-bold'
                  : 'text-slate-300 hover:bg-slate-900'
              }`}
            >
              <span>{item.label}</span>
              {activeTab === item.id && <span className="text-slate-950 font-bold">●</span>}
            </button>
          ))}
          <div className="pt-2">
            <button
              onClick={() => {
                onTabChange('analyze');
                setMobileMenuOpen(false);
              }}
              className="w-full min-h-[44px] py-2.5 rounded-lg bg-cyan-500 text-slate-950 text-xs font-bold text-center flex items-center justify-center gap-2 active:scale-98"
            >
              <span>OPEN WORKSTATION</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </header>
  );
};
